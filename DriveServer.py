#!/opt/anaconda3/bin/python3

"""
DriveServer.py  -  Self-Driving Mercedes: web server front end

A small FastAPI server that drives the car over HTTP, built on the same
car_control library as the DriveCar.py CLI.

Unlike the CLI, the server connects to the BuWizz once at startup and keeps
that connection open, so the steering center is found once (at startup) rather
than before every run. The center is only valid for the connection it was
measured on: if the connection drops, the next request reconnects and re-homes.

Setup / run:
    python3 -m pip install fastapi uvicorn bleak
    python3 DriveServer.py                        # listen on 0.0.0.0:8000
    python3 DriveServer.py --port 8080

Endpoints:
    POST /run                       start a mission with the default duration
    POST /run  {"duration": 60}     same as `DriveCar.py --duration 60`
    POST /recenter                  re-find the steering center now

/run replies 202 straight away and the mission runs in the background; its
progress is printed to this server's console, as with the CLI. /recenter waits
for the re-home to finish (a few seconds). Either replies 409 while the car is
busy with the other (there is only one car).

While idle the server keeps feeding the BuWizz watchdog, which otherwise trips
a few seconds after the last command and drops the BLE connection. Only the
watchdog is fed -- no motor commands -- so the car still stops if this process
dies. Console output is timestamped.
"""

import argparse
import asyncio
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import car_control
from car_control import MISSION_TIME

# The one BuWizz connection, and whether the steering has been homed on it.
car = {"client": None, "char": None, "homed": False}
# Held for anything that moves the car, so runs and recenters never overlap.
car_lock = asyncio.Lock()
# The in-flight mission, if any. Kept so it isn't garbage-collected and so a
# request can be refused while the car is busy.
mission_task: Optional[asyncio.Task] = None


class Timestamped:
    """Wraps a text stream so every line written to it starts with HH:MM:SS."""

    def __init__(self, stream):
        self.stream = stream
        self.at_line_start = True

    def write(self, text):
        out = []
        for line in text.splitlines(keepends=True):
            if self.at_line_start:
                out.append(time.strftime("%H:%M:%S "))
            out.append(line)
            self.at_line_start = line.endswith("\n")
        self.stream.write("".join(out))
        return len(text)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def busy():
    return car_lock.locked() or (mission_task is not None and not mission_task.done())


async def ensure_connected():
    """Connect to the BuWizz if we aren't already. A new connection means a new
    encoder zero, so the old steering center no longer applies."""
    client = car["client"]
    if client is None or not client.is_connected:
        car["client"], car["char"] = await car_control.connect()
        car["homed"] = False


async def recenter():
    """(Re)connect if needed and find the steering center. Caller holds
    car_lock. Raises ConnectionError if the car can't be reached."""
    await ensure_connected()
    car["homed"] = await car_control.home_steering(car["client"], car["char"])
    return car["homed"]


async def _run_mission(duration):
    async with car_lock:
        try:
            # Normally homed at startup; only re-home if the connection was
            # lost or the last attempt failed. Like the CLI, drive even if
            # homing fails.
            if not car["homed"] or not car["client"].is_connected:
                await recenter()
            await car_control.drive_session(car["client"], car["char"],
                                            duration=duration)
        except Exception as e:
            print(f"\nMission failed: {e!r}")


async def keepalive():
    """Feed the watchdog while connected so an idle connection isn't dropped.
    Missions feed it themselves too; the extra feeds are harmless."""
    while True:
        await asyncio.sleep(car_control.WATCHDOG_FEED)
        client = car["client"]
        if client is not None and client.is_connected:
            try:
                await car_control.feed_watchdog(client, car["char"])
            except Exception as e:
                print(f"  [!] keepalive: {e!r}")


@asynccontextmanager
async def lifespan(_app):
    car_control.init_sensor()                      # front obstacle sensor (Pi only)
    async with car_lock:
        try:
            if not await recenter():
                print("Startup: steering homing failed; will retry on the next /run.")
        except ConnectionError as e:
            print(f"Startup: {e} Will retry on the next request.")
    keepalive_task = asyncio.create_task(keepalive())
    yield
    # Shutdown: stop any mission (drive_session stops the motors) and disconnect.
    keepalive_task.cancel()
    if mission_task is not None and not mission_task.done():
        mission_task.cancel()
        await asyncio.gather(mission_task, return_exceptions=True)
    if car["client"] is not None:
        await car_control.disconnect(car["client"])


app = FastAPI(title="Self-Driving Mercedes", lifespan=lifespan)


class RunRequest(BaseModel):
    duration: float = Field(
        MISSION_TIME, gt=0,
        description=f"Mission length in seconds (default {MISSION_TIME:.0f}), "
                    f"same as `DriveCar.py --duration`.")


@app.post("/run", status_code=202)
async def run(req: Optional[RunRequest] = None):
    global mission_task
    req = req or RunRequest()
    if busy():
        raise HTTPException(status_code=409, detail="The car is busy.")
    mission_task = asyncio.create_task(_run_mission(req.duration))
    return {"status": "started", "duration": req.duration}


@app.post("/recenter")
async def recenter_endpoint():
    if busy():
        raise HTTPException(status_code=409, detail="The car is busy.")
    async with car_lock:
        try:
            ok = await recenter()
        except ConnectionError as e:
            raise HTTPException(status_code=503, detail=str(e))
    if not ok:
        raise HTTPException(status_code=500, detail="Steering homing failed.")
    return {"status": "recentered",
            "center": car_control.steer_cal["center"],
            "half": car_control.steer_cal["half"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Web server for the self-driving BuWizz 3.0 car.")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Interface to listen on (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on (default 8000).")
    args = parser.parse_args()
    sys.stdout = Timestamped(sys.stdout)
    sys.stderr = Timestamped(sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
