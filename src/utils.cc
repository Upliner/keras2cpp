#include "utils.h"

namespace keras2cpp {
    Stream::Stream(const std::string& filename)
    : stream_(filename, std::ios::binary) {
        stream_.exceptions();
        if (!stream_.is_open())
            throw std::runtime_error("Cannot open " + filename);
        stream_.seekg(0, std::ios::end);
        size_ = stream_.tellg();
        stream_.seekg(0, std::ios::beg);
    }

    Stream& Stream::reads(char* ptr, size_t count) {
        stream_.read(ptr, static_cast<ptrdiff_t>(count));
        if (!stream_) {
            if (stream_.eof())
                throw KerasException("Corrupt Keras model: unexpected end of file");
            throw std::runtime_error("File read failure");
        }
        return *this;
    }
}
