#include <iostream>
#include <sstream>
#include <string>

class Formatter {
public:
  Formatter() {}
  ~Formatter() {}

  template <typename Type> Formatter &operator<<(const Type &value) {
    stream_ << value;
    return *this;
  }

  std::string str() const { return stream_.str(); }
  operator std::string() const { return 