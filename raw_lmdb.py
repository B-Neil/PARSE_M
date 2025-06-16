import lmdb, gzip, pickle

class RawLMDB:
    def __init__(self, path):
        # open the environment once
        self.env = lmdb.open(path, readonly=True, lock=False)
        # read in the list of keys exactly once, store in a private var
        with self.env.begin() as txn:
            self._keys = [k for k, _ in txn.cursor()]
        self._key2idx = {k: i for i, k in enumerate(self._keys)}

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, idx):
        key = self._keys[idx]
        with self.env.begin() as txn:
            raw = txn.get(key)
        # if val looks gzipped, decompress
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.decompress(raw)
        rec = pickle.loads(raw)
        return rec

    @property
    def ids(self):
        """
        The list of LMDB keys (byte strings).  You can decode() each if
        you need the original filename-like ID.
        """
        return self._keys

    def ids_to_indices(self, id_list):
        """
        Convert iterable of IDs → list of integer indices, preserving order.

        Parameters
        ----------
        id_list : Sequence[str | bytes]
            LMDB keys exactly as they appear in `self.ids`.  Supplying
            a `str` is allowed; it will be UTF-8 encoded before lookup.

        Returns
        -------
        List[int]
            Indices in the same order as `id_list`.

        Raises
        ------
        KeyError
            If any ID is not present in the database.
        """
        idxs = []
        for _id in id_list:
            # ensure bytes for lookup
            key = _id.encode() if isinstance(_id, str) else _id
            try:
                idxs.append(self._key2idx[key])
            except KeyError:
                raise KeyError(f"ID '{_id}' not found in LMDB")
        return idxs



