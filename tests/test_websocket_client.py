from unittest.mock import patch

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy


class _Connection:
    def recv(self):
        return b"metadata"


def test_sync_client_uses_websockets_13_compatible_arguments():
    with patch(
        "wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy.websockets.sync.client.connect",
        return_value=_Connection(),
    ) as connect, patch(
        "wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy.unpackb",
        return_value={},
    ):
        WebsocketClientPolicy(host="127.0.0.1", port=8006)

    assert "ping_interval" not in connect.call_args.kwargs
