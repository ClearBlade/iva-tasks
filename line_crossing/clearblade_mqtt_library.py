import argparse
import json
import logging
import os
import random
import sys
import time

from clearblade.ClearBladeCore import Query, System, cbLogs


class AdapterLibrary:

    DEFAULT_LOG_LEVEL = "info"
    DEFAULT_PLATFORM_URL = "http://localhost:9000"
    DEFAULT_MESSAGING_URL = "localhost:1883"
    DEFAULT_ADAPTER_CONFIG_COLLECTION_NAME = "adapter_config"
    CONNECTED_FLAG = False

    SYSTEM_KEY_ARG_KEY = "CB_SYSTEM_KEY"
    SYSTEM_SECRET_ARG_KEY = "CB_SYSTEM_SECRET"
    DEVICE_NAME_ARG_KEY = "device_name"
    ACTIVE_KEY_ARG_KEY = "active_key"
    SERVICE_ACCOUNT_ARG_KEY = "CB_SERVICE_ACCOUNT"
    SERVICE_ACCOUNT_TOKEN_ARG_KEY = "CB_SERVICE_ACCOUNT_TOKEN"
    PLATFORM_URL_ARG_KEY = "platform_URL"
    MESSAGING_URL_ARG_KEY = "messaging_URL"
    ADAPTER_CONFIG_COLLECTION_NAME_ARG_KEY = "adapter_config_collection_name"
    LOG_LEVEL_ARG_KEY = "log_level"
    SUB_TOPIC_KEY = "sub_topic"

    def __init__(self, adapter_name):
        cbLogs.info(
            "Initializing AdapterLibrary with adapter name: " + adapter_name)
        self.adapter_name = adapter_name
        self._args = {}
        self._cb_system = None
        self._device_client = None
        self._sub_topic = None
        self._cb_message_handler = None

    def parse_arguments(self):
        cbLogs.info(
            "AdapterLibrary - parse_arguments - parsing environment variables and flags")
        self.__parse_env_variables()
        self.__parse_flags()
        self._args[self.LOG_LEVEL_ARG_KEY] = self._args[self.LOG_LEVEL_ARG_KEY].upper()
        logging.basicConfig(level=self._args[self.LOG_LEVEL_ARG_KEY])
        if self._args[self.LOG_LEVEL_ARG_KEY] != "DEBUG":
            cbLogs.DEBUG = False
            cbLogs.MQTT_DEBUG = False
        cbLogs.info(
            "AdapterLibrary - parse_arguments - parsed adapter arguments: " + str(self._args))
        cbLogs.info(
            "AdapterLibrary - parse_arguments - Verifying required adapter arguments")
        if self.SYSTEM_KEY_ARG_KEY not in self._args:
            cbLogs.error("System Key is required, can be supplied with --systemKey flag or " +
                         self.SYSTEM_KEY_ARG_KEY + " environment variable")
            exit(-1)
        if self.SYSTEM_SECRET_ARG_KEY not in self._args:
            cbLogs.error("System Secret is required, can be supplied with --systemSecret flag or " +
                         self.SYSTEM_SECRET_ARG_KEY + " environment variable")
            exit(-1)
        if self.ACTIVE_KEY_ARG_KEY not in self._args and self.SERVICE_ACCOUNT_ARG_KEY not in self._args:
            cbLogs.error(
                "Device Password is required when not using a Service Account, can be supplied with --password flag")
            exit(-1)
        if self.SERVICE_ACCOUNT_ARG_KEY in self._args and self.SERVICE_ACCOUNT_TOKEN_ARG_KEY not in self._args:
            cbLogs.error(
                "Service Account Token is required when a Service Account is specified, this should have automatically been supplied. Check for typos then try again")
            exit(-1)
        cbLogs.info(
            "AdapterLibrary - parse_arguments - Adapter arguments parsed and verified!")

    def initialize_clearblade(self):
        cbLogs.info(
            "AdapterLibrary - initialize_clearblade - initializing with ClearBlade")

        self._cb_system = System(self._args[self.SYSTEM_KEY_ARG_KEY],
                                 self._args[self.SYSTEM_SECRET_ARG_KEY], self._args[self.PLATFORM_URL_ARG_KEY])

        if self.SERVICE_ACCOUNT_ARG_KEY in self._args:
            self.__auth_with_service_account()
        else:
            self.__auth_with_device()

        return self.__fetch_adapter_config()

    def connect_MQTT(self,):
        cbLogs.info(
            "AdapterLibrary - connect_MQTT - Initializing the ClearBlade MQTT message broker")
        self._cb_mqtt = self._cb_system.Messaging(
            self._device_client, client_id=self.adapter_name + "-" + str(random.randint(0, 10000)))
        self._cb_mqtt.on_connect = self.__on_MQTT_connect
        self._cb_mqtt.on_disconnect = self.__on_MQTT_disconnect
        # if topic != "":
        #     self._cb_mqtt.on_subscribe = self.__on_MQTT_subscribe
        #     self._cb_mqtt.on_message = self.__on_MQTT_message_received
        #     self._sub_topic = topic
        self._cb_mqtt.connect()
        # while not self.CONNECTED_FLAG :
        #     time.sleep(1)

    def publish(self, topic, message):
        cbLogs.info(
            "AdapterLibrary - publish - Publishing MQTT message on topic " + topic)
        self._cb_mqtt.publish(topic, message)

    def disconnect_MQTT(self):
        cbLogs.info(
            "AdapterLibrary - disconnect_MQTT - Disconnecting from ClearBlade MQTT message broker")
        print('\n\nDisconnecting MQTT...\n')
        self._cb_mqtt.disconnect()

    def subscribe(self, topic, cb_message_handler):
        cbLogs.info(
            "AdapterLibrary - subscribe - Subscribing to topic " + topic)
        if not self.CONNECTED_FLAG:
            cbLogs.error("""AdapterLibrary - subscribe - MQTT not connected. Verify that the adapter has been initialized and connected to the ClearBlade Platform or Edge before subscribing to a topic""")
            exit(-1)

        self._cb_message_handler = cb_message_handler
        self._cb_mqtt.on_subscribe = self.__on_MQTT_subscribe
        self._cb_mqtt.on_message = self.__on_MQTT_message_received
        self._cb_mqtt.subscribe(topic)

    def unsubscribe(self, topic):
        cbLogs.info(
            "AdapterLibrary - unsubscribe - Unsubscribing from topic " + topic)
        self._cb_mqtt.unsubscribe(topic)

    def __auth_with_service_account(self):
        cbLogs.info(
            "AdapterLibrary - __auth_with_service_account - Authenticating as service account")
        self._device_client = self._cb_system.ServiceUser(
            self._args[self.SERVICE_ACCOUNT_ARG_KEY], self._args[self.SERVICE_ACCOUNT_TOKEN_ARG_KEY])

    def __auth_with_device(self):
        cbLogs.info(
            "AdapterLibrary - __auth_with_device - Authenticating as device")
        self._device_client = self._cb_system.Device(
            self._args[self.DEVICE_NAME_ARG_KEY], self._args[self.ACTIVE_KEY_ARG_KEY])

    def __fetch_adapter_config(self):
        cbLogs.info(
            "AdapterLibrary - __fetch_adapter_config - Retrieving adapter config")

        adapter_config = {
            "adapter_name": self.adapter_name, "adapter_settings": ""}

        collection = self._cb_system.Collection(
            self._device_client, collectionName=self._args[self.ADAPTER_CONFIG_COLLECTION_NAME_ARG_KEY])

        query = Query()
        query.equalTo("adapter_name", self.adapter_name)

        rows = collection.getItems(query)

        if len(rows) == 1:
            if rows[0]["topic_root"] != "":
                adapter_config["topic_root"] = str(rows[0]["topic_root"])
            if rows[0]["adapter_settings"] != "":
                raw_json = json.loads(str(rows[0]["adapter_settings"]))
                # adapter_config["adapter_settings"] = self.__byteify(raw_json)
                adapter_config["adapter_settings"] = raw_json
        else:
            cbLogs.warn("No adapter config found for adapter name " +
                        self.adapter_name + ". Using defaults")

        cbLogs.info(
            "AdapterLibrary - __fetch_adapter_config - Using adapter config: " + str(adapter_config))
        return adapter_config

    def __on_MQTT_connect(self, client, userdata, flags, rc):
        cbLogs.info(
            "AdapterLibrary - __on_MQTT_connect - MQTT successfully connected!")
        # print('\nConnected - Listening....\n')
        self.CONNECTED_FLAG = True
        # print(self._sub_topic)
        # if len(self._sub_topic) != 0:
        #     for topic in self._sub_topic:
        #         client.subscribe(topic)

        # if self._sub_topic != None:
        #     self._cb_mqtt.subscribe(self._sub_topic)

    def __on_MQTT_disconnect(self, client, userdata, rc):
        cbLogs.info(
            "AdapterLibrary - __on_MQTT_disconnect - MQTT disconnected with rc " + str(rc))
        if self._sub_topic != None and rc == 1:
            cbLogs.warn(
                "AdapterLibrary - __on_MQTT_disconnect - Verify that your service account has permission to subscribe to the topic: " + str(self._sub_topic))

    def __on_MQTT_subscribe(self, client, userdata, mid, granted_qos):
        cbLogs.info(
            "AdapterLibrary - __on_MQTT_subscribe - MQTT successfully subcribed to topic " + str(self._sub_topic))

    def __on_MQTT_message_received(self, client, userdata, message):
        cbLogs.info(
            "AdapterLibrary - __on_MQTT_message_received - MQTT message received on topic " + message.topic)
        if self._cb_message_handler != None:
            cbLogs.info("calling message handler")
            self._cb_message_handler(message)

    def __parse_env_variables(self):
        """Parse environment variables"""
        env = os.environ
        possible_vars = [self.SYSTEM_KEY_ARG_KEY, self.SYSTEM_SECRET_ARG_KEY,
                         self.SERVICE_ACCOUNT_ARG_KEY, self.SERVICE_ACCOUNT_TOKEN_ARG_KEY]
        for var in possible_vars:
            if var in env:
                cbLogs.info(
                    "Setting adapter arguments from environment variable: " + var + ": " + str(env[var]))
                self._args[var] = env[var]

    def __parse_flags(self):
        """Parse the command line arguments"""

        parser = argparse.ArgumentParser(description='ClearBlade Adapter')
        parser.add_argument('--systemKey', dest=self.SYSTEM_KEY_ARG_KEY, help='The System Key of the ClearBlade \
                            Plaform "System" the adapter will connect to.')

        parser.add_argument('--systemSecret', dest=self.SYSTEM_SECRET_ARG_KEY, help='The System Secret of the \
                            ClearBlade Plaform "System" the adapter will connect to.')

        parser.add_argument('--deviceName', dest=self.DEVICE_NAME_ARG_KEY, default=self.adapter_name, help='The name of the device that will be used for device \
                            authentication against the ClearBlade Platform or Edge, defined \
                            within the devices table of the ClearBlade platform. The default is ' + self.adapter_name)

        parser.add_argument('--password', dest=self.ACTIVE_KEY_ARG_KEY, help='The password (active key) of the device that will be used for device \
                            authentication against the ClearBlade Platform or Edge, defined within \
                            the devices table of the ClearBlade platform.')

        parser.add_argument('--platformURL', dest=self.PLATFORM_URL_ARG_KEY, default=self.DEFAULT_PLATFORM_URL,
                            help='The HTTP URL of the ClearBlade Platform or Edge the adapter will \
                            connect to (including port if non-standard). The default is ' + self.DEFAULT_PLATFORM_URL)

        parser.add_argument('--messagingURL', dest=self.MESSAGING_URL_ARG_KEY, default=self.DEFAULT_MESSAGING_URL,
                            help='The MQTT URL of the ClearBlade Platform or Edge the adapter will \
                            connect to (including port if non-standard). The default is ' + self.DEFAULT_MESSAGING_URL)

        parser.add_argument('--adapterConfigCollection', dest=self.ADAPTER_CONFIG_COLLECTION_NAME_ARG_KEY,
                            default=self.DEFAULT_ADAPTER_CONFIG_COLLECTION_NAME,
                            help='The name of the ClearBlade Platform data collection which contains \
                            runtime configuration settings for the adapter. The default is ' + self.DEFAULT_ADAPTER_CONFIG_COLLECTION_NAME)

        parser.add_argument('--logLevel', dest=self.LOG_LEVEL_ARG_KEY, default=self.DEFAULT_LOG_LEVEL, choices=['fatal', 'error',
                                                                                                                'warn', 'info', 'debug'], help='The level of logging that \
                            should be utilized by the adapter. The default is ' + self.DEFAULT_LOG_LEVEL)

        parser.add_argument('--subTopic', dest=self.SUB_TOPIC_KEY,
                            help='Topic to subscribe to receive video feed settings')

        args = vars(parser.parse_args(args=sys.argv[1:]))
        for var in args:
            if args[var] != "" and args[var] != None:
                cbLogs.info(
                    "Setting adapter arguments from command line argument: " + var + ": " + str(args[var]))
                self._args[var] = args[var]

    def __byteify(self, input):
        cbLogs.info("in byteify")
        # helper function for python 2.7 to convert unicode to strings in a dict created with json.loads
        # https://stackoverflow.com/a/13105359
        if isinstance(input, dict):
            return {self.__byteify(key): self.__byteify(value)
                    for key, value in input.items()}
        elif isinstance(input, list):
            return [self.__byteify(element) for element in input]
        elif isinstance(input, unicode): # type: ignore
            return input.encode('utf-8')
        else:
            return input

    def get_args(self):
        return self._args