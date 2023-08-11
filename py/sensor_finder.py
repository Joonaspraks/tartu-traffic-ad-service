from c8y_api import CumulocityRestApi
from c8y_api.model import Inventory
from c8y_api.model.managedobjects import DeviceGroup


def get_sensor_list(args):
    devices = args["data_loading"]["devices"]

    source_auth = args["source_auth"]
    source_connection = CumulocityRestApi(
        username=source_auth["username"],
        password=source_auth["password"],
        tenant_id=source_auth["tenant_id"],
        base_url=source_auth["base_url"],
    )
    target_auth = args["target_auth"]
    target_connection = CumulocityRestApi(
        username=target_auth["username"],
        password=target_auth["password"],
        tenant_id=target_auth["tenant_id"],
        base_url=target_auth["base_url"],
    )

    source_inventory = Inventory(c8y=source_connection)

    source_devices_ids = ",".join([str(device["source"]) for device in devices])
    base_query = source_inventory._build_base_query(ids=source_devices_ids)

    source_sensors = list(
        source_inventory._iterate(
            base_query=base_query, limit=9999, parse_func=DeviceGroup.from_json
        )
    )

    assert len(source_sensors) == len(
        devices
    ), "Failed to find a device for each source device id"

    target_inventory = Inventory(c8y=target_connection)
    sensorList = []
    for source_sensor in source_sensors:
        device = list(
            filter(
                lambda device: str(device["source"]) == source_sensor.id,
                devices,
            )
        )[0]
        target_sensor = target_inventory.get(device["target"])
        sensorList.append(
            {
                "name": source_sensor.name,
                "source": source_sensor.id,
                "target": target_sensor.id,
            }
        )
    for sensor in sensorList:
        print(sensor)

    return sensorList
