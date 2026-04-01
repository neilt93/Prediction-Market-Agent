from __future__ import annotations

import base64
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiAuthenticator:
    """Generates RSA-PSS signed headers for Kalshi API requests."""

    def __init__(self, api_key_id: str, private_key_pem: str) -> None:
        self._api_key_id = api_key_id
        self._private_key = serialization.load_pem_private_key(
            private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem,
            password=None,
        )

    @classmethod
    def from_key_file(cls, api_key_id: str, private_key_path: str) -> KalshiAuthenticator:
        with open(private_key_path, "rb") as f:
            pem_data = f.read()
        return cls(api_key_id, pem_data.decode())

    def sign_request(self, method: str, path: str) -> dict[str, str]:
        """Return the three auth headers for a Kalshi API request.

        Args:
            method: HTTP method uppercase (GET, POST, DELETE)
            path: Full path starting with /trade-api/v2/...
        """
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method}{path}"
        signature = self._private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "kalshi-access-key": self._api_key_id,
            "kalshi-access-signature": base64.b64encode(signature).decode(),
            "kalshi-access-timestamp": timestamp,
        }
