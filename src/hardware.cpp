#include "ridge/hardware.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

namespace ridge {

// Datasheet-derived placeholders for A100. Phase 3 calibration replaces these
// with measured values. mmaCyclesPerInst is back-solved so peak FP16 tensor
// throughput is ~312 TFLOPS dense:
//   4096 flops/mma / 2.0 cycles * 1.41e9 Hz * 108 SMs ~= 312e12 FLOP/s.
//
// KNOWN WRONG FOR THE CARD UNDER TEST: hbmBytesPerSec below is the A100 80GB
// figure of about 2.0 TB/s, while every measurement in this project comes from
// an A100-SXM4-40GB, which is about 1.555 TB/s. That makes the HBM roofline
// roughly 29% optimistic. It is left in place on purpose, because changing it
// changes the docs/MODEL.md section 9 worked example and anti-pattern 9 requires
// the derivation to be redone by hand first. Phase 3 replaces this constant with
// a measured value anyway. See PLAN.md Finding 10.
HardwareModel HardwareModel::a100() {
    HardwareModel h;
    h.name = "A100-80GB (datasheet placeholders, replace with calibration)";
    h.numSMs = 108;
    h.clockHz = 1.41e9;
    h.mmaCyclesPerInst = 2.0;
    h.smemBytesPerCycle = 128.0;   // 32 banks * 4 bytes
    h.hbmBytesPerSec = 2.0e12;     // HBM2e ~ 2.0 TB/s, the 80GB part
    h.regsPerSM = 65536;
    h.smemBytesPerSM = 167936;     // 164 KB max shared per SM
    h.maxWarpsPerSM = 64;
    h.maxCtasPerSM = 32;
    h.warpsNeededToHide = 16.0;
    h.calibrated = false;
    return h;
}

namespace {

// A minimal parser for the flat JSON that run-calibration.py writes: one object,
// string and number values, no nesting or arrays.
//
// Written by hand rather than vendoring a library because the schema is a dozen
// scalars and a dependency would be larger than the problem. This does parse
// properly rather than searching for substrings, which matters: the "note" field
// contains prose that mentions several key names, and a substring search would
// happily read a constant out of the middle of a sentence.
class FlatJsonParser {
public:
    explicit FlatJsonParser(const std::string& text) : s_(text) {}

    bool parse(std::map<std::string, std::string>& out, std::string& err) {
        skipSpace();
        if (!expect('{', err)) return false;
        skipSpace();
        if (peek() == '}') return true;

        for (;;) {
            skipSpace();
            std::string key;
            if (!parseString(key, err)) return false;
            skipSpace();
            if (!expect(':', err)) return false;
            skipSpace();

            std::string value;
            if (peek() == '"') {
                if (!parseString(value, err)) return false;
            } else {
                if (!parseBareValue(value, err)) return false;
            }
            out[key] = value;

            skipSpace();
            if (peek() == ',') { ++i_; continue; }
            if (peek() == '}') return true;
            err = "expected ',' or '}' at offset " + std::to_string(i_);
            return false;
        }
    }

private:
    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    void skipSpace() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    bool expect(char c, std::string& err) {
        if (peek() != c) {
            err = std::string("expected '") + c + "' at offset " + std::to_string(i_);
            return false;
        }
        ++i_;
        return true;
    }
    bool parseString(std::string& out, std::string& err) {
        if (!expect('"', err)) return false;
        out.clear();
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) ++i_;
            out.push_back(s_[i_++]);
        }
        return expect('"', err);
    }
    // Numbers, true, false and null all read as an unquoted token. Only numbers
    // appear in this schema, and the caller converts.
    bool parseBareValue(std::string& out, std::string& err) {
        const size_t start = i_;
        while (i_ < s_.size() && s_[i_] != ',' && s_[i_] != '}' &&
               !std::isspace(static_cast<unsigned char>(s_[i_]))) {
            ++i_;
        }
        if (i_ == start) {
            err = "empty value at offset " + std::to_string(start);
            return false;
        }
        out = s_.substr(start, i_ - start);
        return true;
    }

    const std::string& s_;
    size_t i_ = 0;
};

bool takeDouble(const std::map<std::string, std::string>& kv, const char* key,
                double& out, std::string& err) {
    auto it = kv.find(key);
    if (it == kv.end()) {
        err = std::string("missing required key \"") + key + "\"";
        return false;
    }
    char* end = nullptr;
    const double v = std::strtod(it->second.c_str(), &end);
    if (end == it->second.c_str() || *end != '\0') {
        err = std::string("key \"") + key + "\" is not a number: " + it->second;
        return false;
    }
    out = v;
    return true;
}

bool takeInt(const std::map<std::string, std::string>& kv, const char* key,
             int& out, std::string& err) {
    double d = 0.0;
    if (!takeDouble(kv, key, d, err)) return false;
    out = static_cast<int>(d);
    return true;
}

} // namespace

bool HardwareModel::loadFromJson(const std::string& path, HardwareModel& out,
                                 std::string& err) {
    std::ifstream in(path);
    if (!in) {
        err = "cannot open " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    std::map<std::string, std::string> kv;
    FlatJsonParser parser(text);
    if (!parser.parse(kv, err)) {
        err = path + ": " + err;
        return false;
    }

    // Build into a temporary so a failure part way through cannot leave the
    // caller holding a half-calibrated model. See the note in hardware.h.
    HardwareModel h;
    auto name = kv.find("name");
    h.name = name != kv.end() ? name->second : path;

    if (!takeInt(kv, "numSMs", h.numSMs, err) ||
        !takeDouble(kv, "clockHz", h.clockHz, err) ||
        !takeDouble(kv, "mmaCyclesPerInst", h.mmaCyclesPerInst, err) ||
        !takeDouble(kv, "smemBytesPerCycle", h.smemBytesPerCycle, err) ||
        !takeDouble(kv, "hbmBytesPerSec", h.hbmBytesPerSec, err) ||
        !takeInt(kv, "regsPerSM", h.regsPerSM, err) ||
        !takeInt(kv, "smemBytesPerSM", h.smemBytesPerSM, err) ||
        !takeInt(kv, "maxWarpsPerSM", h.maxWarpsPerSM, err) ||
        !takeInt(kv, "maxCtasPerSM", h.maxCtasPerSM, err) ||
        !takeDouble(kv, "warpsNeededToHide", h.warpsNeededToHide, err)) {
        err = path + ": " + err;
        return false;
    }

    // Values that would make the model produce nonsense rather than a wrong
    // number. A zero here would divide by zero or silently zero a ceiling, which
    // is harder to notice than a refusal to load.
    if (h.numSMs <= 0 || h.clockHz <= 0.0 || h.mmaCyclesPerInst <= 0.0 ||
        h.smemBytesPerCycle <= 0.0 || h.hbmBytesPerSec <= 0.0 ||
        h.warpsNeededToHide <= 0.0) {
        err = path + ": a required constant is zero or negative";
        return false;
    }

    h.calibrated = true;
    out = h;
    return true;
}

} // namespace ridge
