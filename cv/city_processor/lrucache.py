class LRUCache:
    def __init__(self, capacity=10):
        self.cache = {}
        self.capacity = capacity
        self.order = []

    def contains(self, key):
        return key in self.cache

    def get(self, key):
        if key in self.cache:
            # Move the accessed key to the end to show that it was recently used
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            # Update the value and move the key to the end to show that it was recently used
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove the oldest item from the cache
                oldest_key = self.order.pop(0)
                del self.cache[oldest_key]
            # Add the new key-value pair to the cache and the order list
            self.cache[key] = value
            self.order.append(key)
