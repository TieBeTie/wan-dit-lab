#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Nodes are supplied in execution order (a topological ordering, with loops
// unrolled). List dependencies before their dependents in each node's uses.
// Analyse uses once; acquire at first use, release at last use.
// CUDA callers must fence work in their release callbacks before destroying it.
class ResourcePlan {
public:
    struct Node {
        std::string name;
        std::vector<std::string> uses;
        std::function<void()> run;
    };

    void resource(std::string name, std::function<void()> acquire,
                  std::function<void()> release) {
        if (!resources_.emplace(std::move(name), Resource{std::move(acquire),
                                                        std::move(release), false}).second)
            throw std::invalid_argument("Duplicate resource");
    }

    void run(const std::vector<Node>& nodes) {
        std::map<std::string, size_t> last;
        std::vector<std::string> acquired;
        // Validate the complete plan before loading anything.
        for (size_t i = 0; i < nodes.size(); ++i)
            for (const auto& name : nodes[i].uses) {
                if (!resources_.count(name))
                    throw std::invalid_argument("Unknown resource: " + name);
                last[name] = i;
            }
        try {
            for (size_t i = 0; i < nodes.size(); ++i) {
                for (const auto& name : nodes[i].uses) {
                    auto& resource = resources_.at(name);
                    if (!resource.live) {
                        // An acquire that partly fails must also be cleaned up.
                        resource.live = true;
                        acquired.push_back(name);
                        resource.acquire();
                    }
                }
                nodes[i].run();
                for (auto name = nodes[i].uses.rbegin(); name != nodes[i].uses.rend(); ++name)
                    if (last.at(*name) == i) release(*name);
            }
        } catch (...) {
            // Preserve the original failure while releasing other live resources.
            for (auto name = acquired.rbegin(); name != acquired.rend(); ++name)
                try { release(*name); } catch (...) {}
            throw;
        }
    }

private:
    struct Resource {
        std::function<void()> acquire, release;
        bool live;
    };
    std::map<std::string, Resource> resources_;

    void release(const std::string& name) {
        auto& resource = resources_.at(name);
        if (resource.live) {
            resource.live = false;
            resource.release();
        }
    }
};
