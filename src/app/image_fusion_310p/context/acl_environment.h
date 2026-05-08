#pragma once

#include <string>

namespace image_fusion_310p
{

class AclEnvironment
{
public:
    static bool acquire(int deviceId, std::string *error);
    static void release(int deviceId);
};

} // namespace image_fusion_310p
