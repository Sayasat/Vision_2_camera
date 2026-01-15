import pyorbbecsdk
ctx = pyorbbecsdk.Context()
devices = ctx.query_devices()
print(f"Найдено устройств: {devices.get_count()}")
for i in range(devices.get_count()):
    dev = devices.get_device_by_index(i)
    info = dev.get_device_info()
    print(f"Устройство {i}: {info.get_name()}, SN: {info.get_serial_number()}")