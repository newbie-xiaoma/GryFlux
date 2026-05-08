#pragma once

#include <string>

namespace yolox_detection_310p
{

class AclEnvironment
{
public:
    static bool acquire(int deviceId, std::string *error);
    static void release(int deviceId);
};

} // namespace yolox_detection_310p
