"""
"""

import atexit
import logging
import os
import select
import tempfile
import threading
import time


class FifoWatcher:
    POLL_RD = select.POLLIN | select.POLLPRI
    POLL_ERR = select.POLLERR | select.POLLHUP | select.POLLNVAL

    def __init__(self, fn, mode, create=False):
        self.fn = fn
        self.mode = mode
        if create:
            if not os.path.exists(self.fn):
                os.mkfifo(self.fn)
        self.fnum = None
        self.poller = select.epoll()
        self.connect()

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        if self.fnum is not None:
            try:
                os.close(self.fnum)
                self.poller.unregister(self.fnum)
            except BrokenPipeError as e:
                logging.info("closing %s resulted in %s", self.fn, e)
        self.fnum = None

    def connected(self):
        return not self.fnum is None

    def connect(self):
        if self.connected():
            return True
        if not os.path.exists(self.fn):
            return False
        try:
            self.fnum = os.open(self.fn, self.mode)
            self.poller.register(self.fnum)
        except OSError:
            self.disconnect()
            return False

        # TODO based on mode, enable/disable read/write

        # TODO flush input?
        #while ffp in select.select([ffp,], [], [], 0.001)[0]:
        #    ffp.read(1)
        return True

    def read(self, block=False, timeout=-1):
        if not self.connected():
            if not self.connect():
                return False
        if block:
            events = self.poller.poll(timeout=timeout)
            if len(events) == 0:
                logging.info("Read timed out")
                return False
            readable = False
            for evt in events:
                fnum, event = evt
                if fnum != self.fnum:
                    continue
                if event & self.POLL_ERR:
                    return False
                if event & self.POLL_RD:
                    readable = True
                    break
            if not readable:
                logging.info("Fifo not readable")
                return False
        try:
            r = os.read(self.fnum, 1)
            return len(r) == 1
        except BrokenPipeError as e:
            logging.info("Reading from %s failed with %s", self.fn, e)
            self.disconnect()
            return False
        except BlockingIOError as e:
            # fifo is in non-blocking mode and no data available
            logging.info("Reading from %s failed with %s", self.fn, e)
            return False

    def write(self):
        if not self.connected():
            if not self.connect():
                return False
        try:
            os.write(self.fnum, b"1")
            # fp.flush()  # TODO how to flush?
        except BrokenPipeError as e:
            logging.info("Writing to %s failed with %s", self.fn, e)
            self.disconnect()
            return False
        return True


def writer(fn, create=False):
    return FifoWatcher(fn, os.O_WRONLY | os.O_NONBLOCK, create=create)


def reader(fn, create=False):
    return FifoWatcher(fn, os.O_RDONLY | os.O_NONBLOCK, create=create)


def test_fifo_watcher():
    tfn = tempfile.mktemp()
    atexit.register(lambda: os.remove(tfn))

    s = writer(tfn, True)
    c = reader(tfn)

    def test_write(s, c):
        assert not c.read(), "First client read did not fail"
        assert s.write(), "First server write failed"
        assert c.read(), "Client failed to read first byte"
        assert not c.read(), "Blank client read did not fail"

    test_write(s, c)

    # test blocking read/write
    t = threading.Thread(target=lambda: c.read(block=True))
    t.start()
    assert t.is_alive(), "Client read thread failed to start"
    time.sleep(0.1)
    assert t.is_alive(), "Client read thread failed to block"
    s.write()
    t.join(0.1)
    assert not t.is_alive(), "Blocking client read failed"

    # test client removal
    del c
    assert not s.write(), "Server write with no client did not fail"

    c = reader(tfn)
    test_write(s, c)

    # test server removal
    del s
    assert not c.read(), "Client read with no server did not fail"
