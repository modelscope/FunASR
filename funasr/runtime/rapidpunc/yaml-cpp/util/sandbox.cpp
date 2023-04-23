#include <iostream>

#include "yaml-cpp/emitterstyle.h"
#include "yaml-cpp/eventhandler.h"
#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

class NullEventHandler : public YAML::EventHandler {
 public:
  using Mark = YAML::Mark;
  using anchor_t = YAML::anchor_t;

  NullEventHandler() = default;

  void OnDocumentStart(const Mark&) override {}
  void OnDocumentEnd() override {}
  void OnNull(const Mark&, anchor_t) override {}
  void OnAlias(const Mark&, anchor_t) override {}
  void OnScalar(const Mark&, const std::string&, anchor_t,
                const std::string&) override {}
  void OnSequenceStart(const Mark&, const std::string&, anchor_t,
                       YAML::EmitterStyle::value style) override {}
  void OnSequenceEnd() override {}
  void OnMapStart(const Mark&, const std::string&, anchor_t,
                  YAML::EmitterStyle::value style) override {}
  void OnMapEnd() override {}
};

int main() {
  YAML::Node root;

  for (;;) {
    YAML::Node node;
    root = node;
  }
  return 0;
}
