#pragma once
auto getDevice(cl::Platform& platform, cl_device_type type, size_t globalMemoryMB);
auto getPlatform(const std::string& vendorNameFilter);