"""WebSocket bridge for VR environment integration.

This module provides a WebSocket server that connects VR applications
to the neural fingerprint stimulation system. VR environments can
trigger sensory experiences and receive real-time feedback.

Protocol:
    - VR sends trigger events when user interacts with virtual objects
    - Bridge initiates corresponding neural stimulation
    - Bridge sends similarity feedback to VR for visual/audio effects
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Set

try:
    import websockets
    from websockets.server import WebSocketServerProtocol

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

from .controller import RealTimeFeedbackController, SystemState

logger = logging.getLogger(__name__)


@dataclass
class VREvent:
    """Event received from VR environment."""

    type: str  # "sensory_trigger", "sensory_stop", "feedback_request"
    modality: Optional[str] = None  # gustatory, olfactory, etc.
    stimulus: Optional[str] = None  # apple_taste, rose_smell, etc.
    intensity: float = 1.0  # 0-1 intensity scale
    duration_s: float = 5.0  # Default duration
    position: Optional[Dict[str, float]] = None  # VR coordinates

    @classmethod
    def from_json(cls, data: Dict) -> "VREvent":
        """Parse event from JSON message."""
        return cls(
            type=data.get("type", "unknown"),
            modality=data.get("modality"),
            stimulus=data.get("stimulus"),
            intensity=data.get("intensity", 1.0),
            duration_s=data.get("duration_s", 5.0),
            position=data.get("position"),
        )


class VRSensoryBridge:
    """WebSocket bridge between VR environment and neural interface.

    Receives sensory event triggers from VR and initiates corresponding
    neural stimulation sequences. Provides real-time similarity feedback
    for VR-side visualization.

    Args:
        feedback_controller: Controller for neural stimulation
        host: Host address for WebSocket server
        port: Port for WebSocket server
    """

    def __init__(
        self,
        feedback_controller: Optional[RealTimeFeedbackController] = None,
        host: str = "localhost",
        port: int = 8765,
    ):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required for VR bridge")

        self.controller = feedback_controller or RealTimeFeedbackController()
        self.host = host
        self.port = port

        self.active_connections: Set[WebSocketServerProtocol] = set()
        self._server = None
        self._running = False

        # Event handlers
        self._event_handlers: Dict[str, Callable] = {
            "sensory_trigger": self._handle_sensory_trigger,
            "sensory_stop": self._handle_sensory_stop,
            "feedback_request": self._handle_feedback_request,
            "ping": self._handle_ping,
        }

    async def start_server(self) -> None:
        """Start the WebSocket server.

        This method blocks until the server is stopped.
        """
        self._running = True

        async with websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
        ) as server:
            self._server = server
            logger.info(f"VR Sensory Bridge listening on ws://{self.host}:{self.port}")
            print(f"VR Sensory Bridge listening on ws://{self.host}:{self.port}")

            # Run until stopped
            while self._running:
                await asyncio.sleep(1)

    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        self._running = False

        # Close all connections
        for ws in list(self.active_connections):
            await ws.close()

        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(
        self,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """Handle incoming WebSocket connection."""
        self.active_connections.add(websocket)
        remote = websocket.remote_address
        logger.info(f"VR client connected: {remote}")

        try:
            # Send welcome message
            await websocket.send(
                json.dumps(
                    {
                        "type": "connected",
                        "message": "Neural Fingerprint System ready",
                        "capabilities": [
                            "gustatory",
                            "olfactory",
                            "tactile",
                            "auditory",
                        ],
                    }
                )
            )

            # Process messages
            async for message in websocket:
                try:
                    event = json.loads(message)
                    await self._process_vr_event(event, websocket)
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps({"type": "error", "message": "Invalid JSON"})
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"VR client disconnected: {remote}")

        finally:
            self.active_connections.discard(websocket)

    async def _process_vr_event(
        self,
        event_data: Dict,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """Process incoming VR event.

        Event types:
        - sensory_trigger: Start sensory stimulation
        - sensory_stop: Stop current stimulation
        - feedback_request: Get current similarity feedback
        - ping: Keep-alive ping
        """
        event = VREvent.from_json(event_data)
        handler = self._event_handlers.get(event.type)

        if handler:
            await handler(event, websocket)
        else:
            await websocket.send(
                json.dumps(
                    {"type": "error", "message": f"Unknown event type: {event.type}"}
                )
            )

    async def _handle_sensory_trigger(
        self,
        event: VREvent,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """Handle sensory trigger event from VR."""
        try:
            if not event.stimulus:
                await websocket.send(
                    json.dumps({"type": "error", "message": "No stimulus specified"})
                )
                return

            # Start neural stimulation
            session_id = await self.controller.start_session(event.stimulus)

            # Send confirmation
            await websocket.send(
                json.dumps(
                    {
                        "type": "stimulation_started",
                        "session_id": session_id,
                        "stimulus": event.stimulus,
                        "modality": event.modality,
                        "intensity": event.intensity,
                    }
                )
            )

            # Start sending periodic feedback updates
            asyncio.create_task(
                self._send_feedback_updates(websocket, session_id, event.duration_s)
            )

        except ValueError as e:
            await websocket.send(
                json.dumps({"type": "error", "message": str(e)})
            )

    async def _handle_sensory_stop(
        self,
        event: VREvent,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """Handle stop stimulation event."""
        await self.controller.stop_session()

        await websocket.send(
            json.dumps({"type": "stimulation_stopped"})
        )

    async def _handle_feedback_request(
        self,
        event: VREvent,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """Handle feedback request event."""
        stats = self.controller.get_session_stats()

        if stats:
            session = self.controller.session
            history = list(session.similarity_history) if session else []

            await websocket.send(
                json.dumps(
                    {
                        "type": "feedback_data",
                        "session_id": stats["session_id"],
                        "state": stats["state"],
                        "similarity": stats["current_similarity"],
                        "avg_similarity": stats["avg_similarity"],
                        "history": history[-10:],  # Last 10 values
                        "elapsed_s": stats["elapsed_s"],
                    }
                )
            )
        else:
            await websocket.send(
                json.dumps(
                    {
                        "type": "feedback_data",
                        "session_id": None,
                        "state": "idle",
                        "similarity": 0.0,
                        "history": [],
                    }
                )
            )

    async def _handle_ping(
        self,
        event: VREvent,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """Handle ping event (keep-alive)."""
        await websocket.send(json.dumps({"type": "pong"}))

    async def _send_feedback_updates(
        self,
        websocket: WebSocketServerProtocol,
        session_id: str,
        duration_s: float,
    ) -> None:
        """Send periodic feedback updates during stimulation."""
        update_interval = 0.1  # 10 Hz
        elapsed = 0.0

        while (
            elapsed < duration_s
            and self.controller.running
            and websocket in self.active_connections
        ):
            stats = self.controller.get_session_stats()

            if stats and stats["session_id"] == session_id:
                session = self.controller.session
                history = list(session.similarity_history) if session else []

                try:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "feedback_update",
                                "similarity": stats["current_similarity"],
                                "avg_similarity": stats["avg_similarity"],
                                "state": stats["state"],
                                "elapsed_s": elapsed,
                                "history": history[-5:],
                            }
                        )
                    )
                except websockets.exceptions.ConnectionClosed:
                    break

            await asyncio.sleep(update_interval)
            elapsed += update_interval

        # Send completion message
        if websocket in self.active_connections:
            try:
                final_stats = self.controller.get_session_stats()
                await websocket.send(
                    json.dumps(
                        {
                            "type": "stimulation_complete",
                            "session_id": session_id,
                            "final_similarity": final_stats["avg_similarity"]
                            if final_stats
                            else 0.0,
                            "duration_s": elapsed,
                        }
                    )
                )
            except websockets.exceptions.ConnectionClosed:
                pass

    async def broadcast_status(self, status: Dict) -> None:
        """Broadcast status update to all connected VR clients.

        Args:
            status: Status dictionary to broadcast
        """
        if not self.active_connections:
            return

        message = json.dumps({"type": "status_broadcast", **status})

        tasks = [ws.send(message) for ws in self.active_connections]
        await asyncio.gather(*tasks, return_exceptions=True)


class VRClient:
    """Test client for VR bridge (for development/testing).

    Args:
        uri: WebSocket URI to connect to
    """

    def __init__(self, uri: str = "ws://localhost:8765"):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required")
        self.uri = uri
        self._ws = None

    async def connect(self) -> None:
        """Connect to the VR bridge server."""
        self._ws = await websockets.connect(self.uri)
        response = await self._ws.recv()
        print(f"Connected: {response}")

    async def trigger_sensory(
        self,
        stimulus: str,
        modality: str = "gustatory",
        intensity: float = 0.8,
        duration_s: float = 5.0,
    ) -> str:
        """Trigger a sensory experience.

        Returns:
            Session ID
        """
        await self._ws.send(
            json.dumps(
                {
                    "type": "sensory_trigger",
                    "modality": modality,
                    "stimulus": stimulus,
                    "intensity": intensity,
                    "duration_s": duration_s,
                }
            )
        )

        response = json.loads(await self._ws.recv())
        return response.get("session_id", "")

    async def stop_sensory(self) -> None:
        """Stop current sensory stimulation."""
        await self._ws.send(json.dumps({"type": "sensory_stop"}))
        await self._ws.recv()

    async def get_feedback(self) -> Dict:
        """Request current feedback data."""
        await self._ws.send(json.dumps({"type": "feedback_request"}))
        response = json.loads(await self._ws.recv())
        return response

    async def listen_updates(
        self,
        callback: Callable[[Dict], None],
        duration_s: float = 10.0,
    ) -> None:
        """Listen for feedback updates.

        Args:
            callback: Function to call with each update
            duration_s: How long to listen
        """
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < duration_s:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(), timeout=1.0
                )
                data = json.loads(message)
                callback(data)
            except asyncio.TimeoutError:
                continue

    async def close(self) -> None:
        """Close the connection."""
        if self._ws:
            await self._ws.close()


async def demo_vr_bridge():
    """Demonstrate the VR bridge with a test server and client."""
    print("Starting VR Bridge Demo...")

    # Create bridge
    controller = RealTimeFeedbackController()
    controller.max_iterations = 10
    bridge = VRSensoryBridge(controller, port=8766)

    # Start server in background
    server_task = asyncio.create_task(bridge.start_server())

    # Wait for server to start
    await asyncio.sleep(1)

    try:
        # Create test client
        client = VRClient("ws://localhost:8766")
        await client.connect()

        # Trigger a sensory experience
        session_id = await client.trigger_sensory(
            stimulus="apple_taste",
            modality="gustatory",
            intensity=0.8,
            duration_s=3.0,
        )
        print(f"Session started: {session_id}")

        # Listen for updates
        def on_update(data: Dict):
            if data.get("type") == "feedback_update":
                print(f"  Similarity: {data.get('similarity', 0):.3f}")
            elif data.get("type") == "stimulation_complete":
                print(f"  Complete! Final: {data.get('final_similarity', 0):.3f}")

        await client.listen_updates(on_update, duration_s=4.0)

        await client.close()

    finally:
        await bridge.stop_server()
        server_task.cancel()

    print("Demo complete!")


if __name__ == "__main__":
    if WEBSOCKETS_AVAILABLE:
        asyncio.run(demo_vr_bridge())
    else:
        print("websockets package not installed. Run: pip install websockets")
