#pragma once

#include <string>

namespace deepsort_track_310p
{

class AclEnvironment
{
public:
    static bool acquire(int deviceId, std::string *error);
    static void release(int deviceId);
};

} // namespace deepsort_track_310p
