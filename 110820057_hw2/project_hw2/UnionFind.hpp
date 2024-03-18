#pragma once

#include <unordered_map>

class UnionFind
{
public:
    void addRoot(const int v);

    int find(const int key);

    int join(const int ka, const int kb);

    std::vector<int> getAllUniqueVal();

private:
    std::unordered_map<int, int> mp;
};