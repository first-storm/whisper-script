# -*- coding: utf-8 -*-
"""
This module contains UI components for the CLI, such as the spinner.
"""

import threading
import time

class Spinner:
    def __init__(self, message='Loading...', spinner_chars='|/-\\'):
        self.message = message
        self.spinner_chars = spinner_chars
        self.spinning = False
        self.thread = None
        self.idx = 0

    def _spin(self):
        while self.spinning:
            print(f'\r{self.spinner_chars[self.idx % len(self.spinner_chars)]} {self.message}', end='', flush=True)
            self.idx += 1
            time.sleep(0.1)

    def start(self):
        if not self.spinning:
            self.spinning = True
            self.thread = threading.Thread(target=self._spin)
            self.thread.start()

    def stop(self, final_message=None):
        if self.spinning:
            self.spinning = False
            if self.thread:
                self.thread.join()
            if final_message:
                print(f'\r✓ {final_message}')
            else:
                print(f'\r✓ {self.message}')

    def fail(self, error_message):
        if self.spinning:
            self.spinning = False
            if self.thread:
                self.thread.join()
        print(f'\r✗ {error_message}')

    def succeed(self, success_message):
        if self.spinning:
            self.spinning = False
            if self.thread:
                self.thread.join()
        print(f'\r✓ {success_message}')