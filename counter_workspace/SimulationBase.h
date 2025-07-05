#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <cstdint>

// Conflict relations between actions
enum class ConflictRelation {
  SequenceBefore = 0,  // SB: First must execute before second
  SequenceAfter = 1,   // SA: First must execute after second
  Conflict = 2,        // C: Cannot execute in same cycle
  ConflictFree = 3     // CF: Can execute in any order
};

// Base class for simulated modules
class SimModule {
public:
  SimModule(const std::string& name) : moduleName(name) {}
  virtual ~SimModule() = default;

  // Register methods
  void registerValueMethod(const std::string& name,
                          std::function<std::vector<int64_t>(const std::vector<int64_t>&)> impl) {
    valueMethods[name] = impl;
  }

  void registerActionMethod(const std::string& name,
                           std::function<void(const std::vector<int64_t>&)> impl) {
    actionMethods[name] = impl;
  }

  void registerRule(const std::string& name, std::function<bool()> impl) {
    rules[name] = impl;
  }

  // Execute methods
  std::vector<int64_t> callValueMethod(const std::string& name, const std::vector<int64_t>& args) {
    auto it = valueMethods.find(name);
    if (it != valueMethods.end()) {
      return it->second(args);
    }
    return {};
  }

  void callActionMethod(const std::string& name, const std::vector<int64_t>& args) {
    auto it = actionMethods.find(name);
    if (it != actionMethods.end()) {
      it->second(args);
    }
  }

  bool canFireRule(const std::string& name) {
    auto it = rules.find(name);
    if (it != rules.end()) {
      return it->second();
    }
    return false;
  }

  const std::string& getName() const { return moduleName; }

protected:
  std::string moduleName;
  std::map<std::string, std::function<std::vector<int64_t>(const std::vector<int64_t>&)>> valueMethods;
  std::map<std::string, std::function<void(const std::vector<int64_t>&)>> actionMethods;
  std::map<std::string, std::function<bool()>> rules;
};

// Main simulation driver
class Simulator {
public:
  Simulator() : cycles(0), verbose(false), dumpStats(false) {}

  void addModule(std::unique_ptr<SimModule> module) {
    modules.push_back(std::move(module));
  }

  void run(int maxCycles);
  void setVerbose(bool v) { verbose = v; }
  void setDumpStats(bool d) { dumpStats = d; }

private:
  std::vector<std::unique_ptr<SimModule>> modules;
  int cycles;
  bool verbose;
  bool dumpStats;
};
