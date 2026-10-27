#include "../scripts/conversion/resource_plan.h"
#include <cassert>
#include <iostream>

int main() {
    std::vector<std::string> events;
    ResourcePlan plan;
    for (const std::string name : {"t5", "dit", "vae"})
        plan.resource(name, [&, name] { events.push_back("load " + name); },
                           [&, name] { events.push_back("free " + name); });
    plan.run({{"encode", {"t5"}, [&] { events.push_back("encode"); }},
              {"chunk0", {"dit"}, [&] { events.push_back("chunk0"); }},
              {"decode0", {"vae"}, [&] { events.push_back("decode0"); }},
              {"chunk1", {"dit"}, [&] { events.push_back("chunk1"); }},
              {"decode1", {"vae"}, [&] { events.push_back("decode1"); }}});
    assert((events == std::vector<std::string>{
        "load t5", "encode", "free t5", "load dit", "chunk0", "load vae", "decode0",
        "chunk1", "free dit", "decode1", "free vae"}));

    events.clear();
    ResourcePlan failing;
    failing.resource("mapping", [] {}, [&] { events.push_back("mapping"); });
    failing.resource("worker", [] {}, [&] { events.push_back("worker"); });
    try {
        failing.run({{"fail", {"mapping", "worker"}, [] { throw std::runtime_error("kernel"); }}});
        assert(false);
    } catch (const std::runtime_error& error) { assert(std::string(error.what()) == "kernel"); }
    assert((events == std::vector<std::string>{"worker", "mapping"}));

    events.clear();
    ResourcePlan invalid;
    invalid.resource("known", [&] { events.push_back("unexpected load"); }, [] {});
    try {
        invalid.run({{"first", {"known"}, [] {}}, {"invalid", {"missing"}, [] {}}});
        assert(false);
    } catch (const std::invalid_argument&) {}
    assert(events.empty());

    bool cleaned = false;
    ResourcePlan partial;
    partial.resource("partial", [] { throw std::runtime_error("allocation"); }, [&] { cleaned = true; });
    try { partial.run({{"allocate", {"partial"}, [] {}}}); }
    catch (const std::runtime_error&) {}
    assert(cleaned);
    std::cout << "resource lifetime tests passed\n";
}
