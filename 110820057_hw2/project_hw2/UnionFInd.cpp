#include "UnionFind.hpp"
#include <cassert>
#include <unordered_set>

void UnionFind::addRoot(const int v)
{
    assert(mp.find(v) == mp.end());
    mp[v] = v;
}

int UnionFind::find(const int key)
{
    assert(mp.find(key) != mp.end());
    if (key != mp[key]) {
        return mp[key] = find(mp[key]);
    }
    return key;
}

int UnionFind::join(const int ka, const int kb)
{
    const auto ra = find(ka);
    const auto rb = find(kb);
    if (ra < rb) {
        mp[rb] = ra;
        return ra;
    }
    mp[ra] = rb;
    return rb;
}

std::vector<int> UnionFind::getAllUniqueVal()
{
    std::unordered_set<int> us;
    for (auto& it : mp) {
        us.insert(it.second);
    }
    return { us.begin(), us.end() };
}
