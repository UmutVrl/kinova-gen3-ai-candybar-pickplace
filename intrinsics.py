#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time

from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2

#
# Dictionary of all Sensor strings
#
all_sensor_strings = {
    VisionConfig_pb2.SENSOR_UNSPECIFIED: "Unspecified sensor",
    VisionConfig_pb2.SENSOR_COLOR: "Color",
    VisionConfig_pb2.SENSOR_DEPTH: "Depth"
}

#
# Dictionary of all Resolution strings
#
all_resolution_strings = {
    VisionConfig_pb2.RESOLUTION_UNSPECIFIED: "Unspecified resolution",
    VisionConfig_pb2.RESOLUTION_320x240: "320x240",
    VisionConfig_pb2.RESOLUTION_424x240: "424x240",
    VisionConfig_pb2.RESOLUTION_480x270: "480x270",
    VisionConfig_pb2.RESOLUTION_640x480: "640x480",
    VisionConfig_pb2.RESOLUTION_1280x720: "1280x720",
    VisionConfig_pb2.RESOLUTION_1920x1080: "1920x1080"
}

# Lists of supported options for a sensor
supported_color_options = []
supported_depth_options = []


#
# Returns the device identifier of the Vision module, 0 if not found
#
def vision_get_device_id(device_manager):
    vision_device_id = 0

    # Getting all device routing information (from DeviceManagerClient service)
    all_devices_info = device_manager.ReadAllDevices()

    vision_handles = [hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION]
    if len(vision_handles) == 0:
        print("Error: there is no vision device registered in the devices info")
    elif len(vision_handles) > 1:
        print("Error: there are more than one vision device registered in the devices info")
    else:
        handle = vision_handles[0]
        vision_device_id = handle.device_identifier
        print("Vision module found, device Id: {0}".format(vision_device_id))

    return vision_device_id


#
# Display how to use these examples
#
def display_usage():
    print("\n")
    print("\t######################################################################################")
    print("\t# To use these examples:                                                             #")
    print("\t#  - Connect to the robot's web page                                                 #")
    print("\t#  - Select the Camera view page                                                     #")
    print("\t#  - Observe the effects of changing the Color sensor options                        #")
    print("\t######################################################################################")
    print("\n")


#
# Get the name of a sensor
#
def get_sensor_name(sensor):
    if sensor == VisionConfig_pb2.SENSOR_COLOR:
        name = "COLOR"
    elif sensor == VisionConfig_pb2.SENSOR_DEPTH:
        name = "DEPTH"
    else:
        name = "***UNSUPPORTED***"
    return name


#
# Get the list of supported options for a sensor
# Each list item is a dictionary describing an option information
#
def get_sensor_supported_options(sensor):
    if sensor == VisionConfig_pb2.SENSOR_COLOR:
        supported_option = supported_color_options
    elif sensor == VisionConfig_pb2.SENSOR_DEPTH:
        supported_option = supported_depth_options
    else:
        supported_option = []
    return supported_option


#
# Display the information of a specific sensor option
#
def display_sensor_option_information(option_info):
    print("Option id: {0:02d}  name: {1}  is_writable: {2}\n"
          "minimum: {3:0.06f}  maximum: {4:0.06f}\n"
          "   step: {5:0.06f}  default: {6:0.06f}\n"
          .format(option_info['id'], option_info['name'], option_info['writable'],
                  option_info['min'], option_info['max'], option_info['step'], option_info['default']))


#
# Add option information to a list of supported options for a sensor
# The added item is a dictionary describing the option information, with the following fields:
#   'id', 'name', 'writable', 'min', 'max', 'step', 'default'
# Then, display the option information
#
def add_and_display_sensor_supported_option(option_info):
    display_option_info = True
    option_info_dict = {'id': option_info.option, 'name': VisionConfig_pb2.Option.Name(option_info.option),
                        'writable': True if int(option_info.read_only) == 0 else False, 'min': option_info.minimum,
                        'max': option_info.maximum, 'step': option_info.step, 'default': option_info.default_value}

    if option_info.sensor == VisionConfig_pb2.SENSOR_COLOR:
        supported_color_options.append(option_info_dict)
    elif option_info.sensor == VisionConfig_pb2.SENSOR_DEPTH:
        supported_depth_options.append(option_info_dict)
    else:
        print("Unsupported sensor {0} for option id {1}, not adding to any list!".format(option_info.sensor,
                                                                                         option_info.option))
        display_option_info = False

    # Display option information
    if display_option_info:
        display_sensor_option_information(option_info_dict)


#
# For all sensor options, set their value based on the specified range
#
def set_sensor_options_values_by_range(sensor, value_range, vision_config, vision_device_id):
    option_value = VisionConfig_pb2.OptionValue()
    sensor_options = get_sensor_supported_options(sensor)
    option_value.sensor = sensor

    for option in sensor_options:
        if option['writable']:
            try:
                option_value.option = option['id']
                option_value.value = option[value_range]
                vision_config.SetOptionValue(option_value, vision_device_id)
                print("Set value ({0:0.06f}) for option '{1}'".format(option_value.value, option['name']))
            except Exception as ex:
                print("Failed to set {0} value for option '{1}': {2}".format(value_range, option['name'], str(ex)))


#
# For all sensor options, validate their value based on the specified range
#
def validate_sensor_options_values_by_range(sensor, value_range, vision_config, vision_device_id):
    option_identifier = VisionConfig_pb2.OptionIdentifier()
    option_value_reply = VisionConfig_pb2.OptionValue()
    sensor_options = get_sensor_supported_options(sensor)
    option_identifier.sensor = sensor

    for option in sensor_options:
        if option['writable']:
            try:
                option_identifier.option = option['id']
                option_value_reply = vision_config.GetOptionValue(option_identifier, vision_device_id)
                print("Confirm received value ({0:0.06f}) for option '{1}' --> {2}" \
                      .format(option_value_reply.value, option['name'],
                              "OK" if option_value_reply.value == option[value_range] else "*** FAILED ***"))
            except Exception as ex:
                print("Failed to get value for option '{0}': {1}".format(option['name'], str(ex)))


# Example showing how to get the sensors options information
# Note: This function must be called in order to fill up the lists of sensors supported options
def routed_vision_get_option_information(vision_config, vision_device_id):
    print("\n** Example showing how to get the sensors options information **")
    sensors = VisionConfig_pb2.Sensor.values()[1:]
    options = VisionConfig_pb2.Option.values()[1:]
    option_identifier = VisionConfig_pb2.OptionIdentifier()

    # For all sensors, determine which options are supported and populate specific list
    for sensor in sensors:
        option_identifier.sensor = sensor
        sensor_name = get_sensor_name(sensor)
        print("\n-- Using Vision Config Service to get information for all {0} sensor options --".format(sensor_name))
        for option in options:
            try:
                option_identifier.option = option
                option_info = vision_config.GetOptionInformation(option_identifier, vision_device_id)
                if option_info.sensor == sensor and option_info.option == option:
                    if int(option_info.supported) == 1:
                        add_and_display_sensor_supported_option(option_info)
                else:
                    print("Unexpected mismatch of sensor or option in returned information for option id {0}!".format(
                        option))
            except Exception:
                # The option is simply not supported
                pass


#
# Example showing how to get the sensors options values
#
def routed_vision_get_sensor_options_values(vision_config, vision_device_id):
    print("\n** Example showing how to get the sensors options values **")
    sensors = VisionConfig_pb2.Sensor.values()[1:]
    option_identifier = VisionConfig_pb2.OptionIdentifier()

    # For all sensors, get their supported options value
    for sensor in sensors:
        option_identifier.sensor = sensor
        sensor_name = get_sensor_name(sensor)
        sensor_options = get_sensor_supported_options(sensor)
        print("\n-- Using Vision Config Service to get value for all {0} sensor options --".format(sensor_name))
        for option in sensor_options:
            try:
                option_identifier.option = option['id']
                option_value = vision_config.GetOptionValue(option_identifier, vision_device_id)
                print("Option '{0}' has value {1:0.06f}".format(option['name'], option_value.value))
            except Exception as ex:
                print("Failed to get value of option '{0}': {1}".format(option['name'], str(ex)))
        print("")


#
# Example showing how to set the sensors options values
#
def routed_vision_set_sensor_options_values(vision_config, vision_device_id):
    print("\n** Example showing how to set the sensors options values **")
    sensors = VisionConfig_pb2.Sensor.values()[1:]

    # For all sensors, set and confirm options values
    for value_range in ['max', 'default']:
        for sensor in sensors:
            sensor_name = get_sensor_name(sensor)

            print("\n-- Using Vision Config Service to set {0} value for all {1} sensor options --".format(value_range,
                                                                                                           sensor_name))
            set_sensor_options_values_by_range(sensor, value_range, vision_config, vision_device_id)

            print("\n-- Using Vision Config Service to confirm {0} value was set for all {1} sensor options --".format(
                value_range, sensor_name))
            validate_sensor_options_values_by_range(sensor, value_range, vision_config, vision_device_id)

            if sensor == VisionConfig_pb2.SENSOR_COLOR:
                print("\n-- Waiting for 5 seconds to observe the effects of the new COLOR sensor options values... --")
                time.sleep(5)
            else:
                print("")


#
# Example confirming that sensors options values are restored upon a reboot of the Vision module
#
def routed_vision_confirm_saved_sensor_options_values(vision_config, device_config, vision_device_id):
    print("\n** Example confirming that sensors options values are restored upon a reboot of the Vision module **")
    sensors = VisionConfig_pb2.Sensor.values()[1:]

    # For all sensors, set and confirm options values
    for value_range in ['min', 'default']:
        for sensor in sensors:
            sensor_name = get_sensor_name(sensor)

            print("\n-- Using Vision Config Service to set {0} value for all {1} sensor options --".format(value_range,
                                                                                                           sensor_name))
            set_sensor_options_values_by_range(sensor, value_range, vision_config, vision_device_id)

            print("\n-- Using Vision Config Service to confirm {0} value was set for all {1} sensor options --".format(
                value_range, sensor_name))
            validate_sensor_options_values_by_range(sensor, value_range, vision_config, vision_device_id)

        # If we just set the options' minimum value, reboot the Vision module device
        if value_range == "min":
            # Reboot with a delay
            delay_to_reboot_ms = 5000
            reboot_request = DeviceConfig_pb2.RebootRqst()
            reboot_request.delay = delay_to_reboot_ms
            print(
                "\n-- Using Device Config Service to reboot the Vision module in {0} milliseconds. Please wait... --".format(
                    delay_to_reboot_ms))
            device_config.RebootRequest(reboot_request, vision_device_id)

            # Wait until the Vision module is rebooted completely
            wait_after_reboot_sec = 35 + (delay_to_reboot_ms / 1000)
            time.sleep(wait_after_reboot_sec)

            # For all sensors, confirm their option values were restored
            for sensor in sensors:
                sensor_name = get_sensor_name(sensor)

                print(
                    "\n-- Using Vision Config Service to confirm {0} value was restored after reboot for all {1} "
                    "sensor options --".format(
                        value_range, sensor_name))
                validate_sensor_options_values_by_range(sensor, value_range, vision_config, vision_device_id)
        else:
            print("")


#
# Returns a string matching the requested sensor
#
def sensor_to_string(sensor):
    return all_sensor_strings.get(sensor, "Unknown sensor")


#
# Returns a string matching the requested resolution
#
def resolution_to_string(resolution):
    return all_resolution_strings.get(resolution, "Unknown resolution")


#
# Prints the intrinsic parameters on stdout
#
def print_intrinsic_parameters(intrinsics):
    print("Sensor: {0} ({1})".format(intrinsics.sensor, sensor_to_string(intrinsics.sensor)))
    print("Resolution: {0} ({1})".format(intrinsics.resolution, resolution_to_string(intrinsics.resolution)))
    print("Principal point x: {0:.6f}".format(intrinsics.principal_point_x))
    print("Principal point y: {0:.6f}".format(intrinsics.principal_point_y))
    print("Focal length x: {0:.6f}".format(intrinsics.focal_length_x))
    print("Focal length y: {0:.6f}".format(intrinsics.focal_length_y))
    print("Distortion coefficients: [{0:.6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f}]".format(
        intrinsics.distortion_coeffs.k1,
        intrinsics.distortion_coeffs.k2,
        intrinsics.distortion_coeffs.p1,
        intrinsics.distortion_coeffs.p2,
        intrinsics.distortion_coeffs.k3))


#
# Example showing how to retrieve the intrinsic parameters of the Color and Depth sensors
#
def get_routed_vision_intrinsics(vision_config, vision_device_id):
    sensor_id = VisionConfig_pb2.SensorIdentifier()
    profile_id = VisionConfig_pb2.IntrinsicProfileIdentifier()

    print("\n\n** Getting intrinsic parameters of the Color and Depth sensors **")

    print("\n-- Getting intrinsic parameters of active color resolution --")
    sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
    intrinsics = vision_config.GetIntrinsicParameters(sensor_id, vision_device_id)
    print_intrinsic_parameters(intrinsics)

    print("\n-- Getting intrinsic parameters of active depth resolution --")
    sensor_id.sensor = VisionConfig_pb2.SENSOR_DEPTH
    intrinsics = vision_config.GetIntrinsicParameters(sensor_id, vision_device_id)
    print_intrinsic_parameters(intrinsics)

    print("\n-- Getting intrinsic parameters for color resolution 1280x720 --")
    profile_id.sensor = VisionConfig_pb2.SENSOR_COLOR
    profile_id.resolution = VisionConfig_pb2.RESOLUTION_1280x720
    intrinsics = vision_config.GetIntrinsicParametersProfile(profile_id, vision_device_id)
    print_intrinsic_parameters(intrinsics)

    print("\n-- Getting intrinsic parameters for depth resolution 424x240 --")
    profile_id.sensor = VisionConfig_pb2.SENSOR_DEPTH
    profile_id.resolution = VisionConfig_pb2.RESOLUTION_424x240
    intrinsics = vision_config.GetIntrinsicParametersProfile(profile_id, vision_device_id)
    print_intrinsic_parameters(intrinsics)


def get_active_routed_vision_intrinsics(vision_config, vision_device_id):
    sensor_id = VisionConfig_pb2.SensorIdentifier()
    profile_id = VisionConfig_pb2.IntrinsicProfileIdentifier()

    print("\n-- Getting intrinsic parameters of active color resolution --")
    sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
    intrinsics = vision_config.GetIntrinsicParameters(sensor_id, vision_device_id)
    print_intrinsic_parameters(intrinsics)

    # Extract width and height
    # width = intrinsics.resolution.width
    # height = intrinsics.resolution.height

    return intrinsics


#
# Example showing how to set the intrinsic parameters of the Color and Depth sensors
#
def routed_vision_set_intrinsics(vision_config, vision_device_id):
    profile_id = VisionConfig_pb2.IntrinsicProfileIdentifier()
    intrinsics_new = VisionConfig_pb2.IntrinsicParameters()

    print("\n-- Using Vision Config Service to get current intrinsic parameters for color resolution 1280x720 --")
    profile_id.sensor = VisionConfig_pb2.SENSOR_COLOR
    profile_id.resolution = VisionConfig_pb2.RESOLUTION_1280x720
    intrinsics_old = vision_config.GetIntrinsicParametersProfile(profile_id, vision_device_id)
    print_intrinsic_parameters(intrinsics_old)

    print("\n-- Using Vision Config Service to set new intrinsic parameters for color resolution 1280x720 --")
    intrinsics_new.sensor = profile_id.sensor
    intrinsics_new.resolution = profile_id.resolution
    # Old values
    # intrinsics_new.principal_point_x = 640 / 2 + 0.123456
    # intrinsics_new.principal_point_y = 480 / 2 + 1.789012
    # intrinsics_new.focal_length_x = 650.567890
    # intrinsics_new.focal_length_y = 651.112233
    # intrinsics_new.distortion_coeffs.k1 = 0.2
    # intrinsics_new.distortion_coeffs.k2 = 0.05
    # intrinsics_new.distortion_coeffs.p1 = 1.2
    # intrinsics_new.distortion_coeffs.p2 = 0.999999
    # intrinsics_new.distortion_coeffs.k3 = 0.001

    # Camera matrix values from your calibration
    intrinsics_new.focal_length_x = 1324.58038  # matrix[0,0]
    intrinsics_new.focal_length_y = 1320.97766  # matrix[1,1]
    intrinsics_new.principal_point_x = 645.196528  # matrix[0,2]
    intrinsics_new.principal_point_y = 225.297781  # matrix[1,2]

    # Distortion coefficients from your calibration (OpenCV order: k1, k2, p1, p2, k3)
    intrinsics_new.distortion_coeffs.k1 = -0.00994774168  # dist[0]
    intrinsics_new.distortion_coeffs.k2 = 0.281289667  # dist[1]
    intrinsics_new.distortion_coeffs.p1 = -0.000934587029  # dist[2]
    intrinsics_new.distortion_coeffs.p2 = -0.00489698821  # dist[3]
    intrinsics_new.distortion_coeffs.k3 = -1.21679259  # dist[4]

    vision_config.SetIntrinsicParameters(intrinsics_new, vision_device_id)
    print("\n** Set! **")

    # print("\n-- Using Vision Config Service to get new intrinsic parameters for color resolution 1280x720 --")
    intrinsics_reply = vision_config.GetIntrinsicParametersProfile(profile_id, vision_device_id)
    # print_intrinsic_parameters(intrinsics_reply)

    # print("\n-- Using Vision Config Service to set back old intrinsic parameters for color resolution 640x480 --")
    # vision_config.SetIntrinsicParameters(intrinsics_old, vision_device_id)


#
# Example showing how to set the intrinsic parameters of the Color and Depth sensors
#
def example_routed_vision_set_intrinsics(vision_config, vision_device_id):
    profile_id = VisionConfig_pb2.IntrinsicProfileIdentifier()
    intrinsics_new = VisionConfig_pb2.IntrinsicParameters()

    print("\n\n** Example showing how to set the intrinsic parameters of the Color and Depth sensors **")

    print("\n-- Using Vision Config Service to get current intrinsic parameters for color resolution 640x480 --")
    profile_id.sensor = VisionConfig_pb2.SENSOR_COLOR
    profile_id.resolution = VisionConfig_pb2.RESOLUTION_640x480
    intrinsics_old = vision_config.GetIntrinsicParametersProfile(profile_id, vision_device_id)
    print_intrinsic_parameters(intrinsics_old)

    print("\n-- Using Vision Config Service to set new intrinsic parameters for color resolution 640x480 --")
    intrinsics_new.sensor = profile_id.sensor
    intrinsics_new.resolution = profile_id.resolution
    intrinsics_new.principal_point_x = 640 / 2 + 0.123456
    intrinsics_new.principal_point_y = 480 / 2 + 1.789012
    intrinsics_new.focal_length_x = 650.567890
    intrinsics_new.focal_length_y = 651.112233
    intrinsics_new.distortion_coeffs.k1 = 0.2
    intrinsics_new.distortion_coeffs.k2 = 0.05
    intrinsics_new.distortion_coeffs.p1 = 1.2
    intrinsics_new.distortion_coeffs.p2 = 0.999999
    intrinsics_new.distortion_coeffs.k3 = 0.001
    vision_config.SetIntrinsicParameters(intrinsics_new, vision_device_id)

    print("\n-- Using Vision Config Service to get new intrinsic parameters for color resolution 640x480 --")
    intrinsics_reply = vision_config.GetIntrinsicParametersProfile(profile_id, vision_device_id)
    print_intrinsic_parameters(intrinsics_reply)

    print("\n-- Using Vision Config Service to set back old intrinsic parameters for color resolution 640x480 --")
    vision_config.SetIntrinsicParameters(intrinsics_old, vision_device_id)

    print("\n-- Using Vision Config Service to get current intrinsic parameters for depth resolution 424x240 --")
    profile_id.sensor = VisionConfig_pb2.SENSOR_DEPTH
    profile_id.resolution = VisionConfig_pb2.RESOLUTION_424x240
    intrinsics_old = vision_config.GetIntrinsicParametersProfile(profile_id, vision_device_id)
    print_intrinsic_parameters(intrinsics_old)

    print("\n-- Using Vision Config Service to set new intrinsic parameters for depth resolution 424x240 --")
    intrinsics_new.sensor = profile_id.sensor
    intrinsics_new.resolution = profile_id.resolution
    intrinsics_new.principal_point_x = 424 / 2 + 0.123456
    intrinsics_new.principal_point_y = 240 / 2 + 1.789012
    intrinsics_new.focal_length_x = 315.567890
    intrinsics_new.focal_length_y = 317.112233
    intrinsics_new.distortion_coeffs.k1 = 0.425
    intrinsics_new.distortion_coeffs.k2 = 1.735102
    intrinsics_new.distortion_coeffs.p1 = 0.1452
    intrinsics_new.distortion_coeffs.p2 = 0.767574
    intrinsics_new.distortion_coeffs.k3 = 2.345678
    vision_config.SetIntrinsicParameters(intrinsics_new, vision_device_id)

    print("\n-- Using Vision Config Service to get new intrinsic parameters for depth resolution 424x240 --")
    intrinsics_reply = vision_config.GetIntrinsicParametersProfile(profile_id, vision_device_id)
    print_intrinsic_parameters(intrinsics_reply)

    print("\n-- Using Vision Config Service to set back old intrinsic parameters for depth resolution 424x240 --")
    vision_config.SetIntrinsicParameters(intrinsics_old, vision_device_id)


def get_vision_device_id(device_manager):
    vision_device_id = 0

    all_devices_info = device_manager.ReadAllDevices()

    vision_handles = [hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION]
    if len(vision_handles) == 0:
        print("Error: there is no vision device registered in the devices info")
    elif len(vision_handles) > 1:
        print("Error: there are more than one vision device registered in the devices info")
    else:
        handle = vision_handles[0]
        vision_device_id = handle.device_identifier
        print("Vision module found, device Id: {0}".format(vision_device_id))

    return vision_device_id


def disable_autofocus(vision_config, vision_device_id):
    sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
    sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_DISABLE_FOCUS
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print("Autofocus disabled")


#
# Wait for 10 seconds, allowing to see the effects of the focus action
#
def wait_for_focus_action():
    print("-- Waiting for 10 seconds to observe the effects of the focus action... --")
    time.sleep(10)


#
# Example showing how to play with the auto-focus of the Color camera
#
def routed_vision_do_autofocus_action(vision_config, vision_device_id):
    print("\n** Example showing how to play with the auto-focus of the Color camera **")
    sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
    sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR

    print("\n-- Using Vision Config Service to disable the auto-focus --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_DISABLE_FOCUS
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print("-- Place or remove an object from the center of the camera, observe the focus doesn't change --")
    wait_for_focus_action()

    print("\n-- Using Vision Config Service to enable the auto-focus --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_START_CONTINUOUS_FOCUS
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print("-- Place an object in the center of the camera, observe the focus adjusts automatically --")
    wait_for_focus_action()

    print("\n-- Using Vision Config Service to pause the auto-focus --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_PAUSE_CONTINUOUS_FOCUS
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print(
        "-- Move the object away from the center of the camera and then back, but at a different distance, "
        "observe the focus doesn't change --")
    wait_for_focus_action()

    print("\n-- Using Vision Config Service to focus now --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_FOCUS_NOW
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print("-- Observe the focus tried to adjust to the object in front to the camera --")
    wait_for_focus_action()

    print("\n-- Using Vision Config Service to re-enable the auto-focus --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_START_CONTINUOUS_FOCUS
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print(
        "-- Move the object away from the center of the camera and then back, but at a different distance, "
        "observe the focus adjusts automatically --")
    wait_for_focus_action()


#
# Example showing how to set the focus of the Color camera to a X-Y point in the camera image
#
def routed_vision_set_focus_point(vision_config, vision_device_id):
    print("\n** Example showing how to set the focus of the Color camera to a X-Y point in the camera image **")
    sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
    sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR

    print(
        "\n-- Using Vision Config Service to set the focus point in the center of the lower right quadrant of the camera image --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_FOCUS_POINT
    sensor_focus_action.focus_point.x = int(1280 * 3 / 4)
    sensor_focus_action.focus_point.y = int(720 * 3 / 4)
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_FOCUS_NOW
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print(
        "-- Place an object in the center of the lower right quadrant of the camera image, observe the object gets into focus --")
    wait_for_focus_action()

    print("\n-- Using Vision Config Service to set the focus point back in the middle the camera image--")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_FOCUS_POINT
    sensor_focus_action.focus_point.x = int(1280 / 2)
    sensor_focus_action.focus_point.y = int(720 / 2)
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_FOCUS_NOW
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print("-- Place an object in the center of the camera image, observe the object gets into focus --")
    wait_for_focus_action()


#
# Example showing how to set the manual focus of the Color camera (changes the focus distance)
#
def routed_vision_set_manual_focus(vision_config, vision_device_id):
    print("\n** Example showing how to set the manual focus of the Color camera (changes the focus distance) **")
    sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
    sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR

    print("\n-- Using Vision Config Service to set the manual focus on a very close object (close-up view) --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
    sensor_focus_action.manual_focus.value = 1023  # Maximum accepted value
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print("-- Place an object at around 2 inches away from the center of the camera, observe the object is in focus --")
    wait_for_focus_action()

    print("\n-- Using Vision Config Service to set the manual focus on an object at a greater distance --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
    sensor_focus_action.manual_focus.value = 0  # Mininum accepted value
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print("-- Move the object away from the camera until it gets into focus --")
    wait_for_focus_action()

    print("\n-- Using Vision Config Service to set the manual focus on a relatively close object (normal view) --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
    sensor_focus_action.manual_focus.value = 350
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print("-- Move the object at around 8 inches away from the center of the camera, observe the object is in focus --")
    wait_for_focus_action()

    print("\n-- Using Vision Config Service to re-enable the auto-focus --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_START_CONTINUOUS_FOCUS
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    print(
        "-- Move the object away from the camera and then back, but at a different distance, observe the focus "
        "adjusts automatically --")
    wait_for_focus_action()


# for 35 cm away table
def routed_vision_set_manual_focus_medium_distance(vision_config, vision_device_id):
    print("\n** the manual focus of the Color camera (changes the focus distance to 13.7 inches / 35 cm) **")
    sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
    sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR

    print("\n-- Using Vision Config Service to set the manual focus on a relatively close object (normal view) --")
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
    sensor_focus_action.manual_focus.value = 216
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)

    # print("-- Move the object at around 13.7 inches / 35 cm away from the center of the camera, observe the object is "
    #      "in focus --")
    # wait_for_focus_action()

    def main():
        # Import the utilities helper module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import utilities

        # Parse arguments
        args = utilities.parseConnectionArguments()

        # Create connection to the device and get the router
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            device_manager = DeviceManagerClient(router)
            device_config = DeviceConfigClient(router)
            vision_config = VisionConfigClient(router)

            # example core
            vision_device_id = vision_get_device_id(device_manager)

            if vision_device_id != 0:
                pass
                # routed_vision_get_option_information(vision_config, vision_device_id)
                # routed_vision_get_sensor_options_values(vision_config, vision_device_id)
                # routed_vision_set_sensor_options_values(vision_config, vision_device_id)
                # routed_vision_confirm_saved_sensor_options_values(vision_config, device_config,vision_device_id)
