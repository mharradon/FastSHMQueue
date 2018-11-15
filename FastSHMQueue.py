from multiprocessing import Manager, Process, Pool, Array
from array import array
from functools import reduce
import threading
import numpy as np
import queue
import time

test_size = (100,100,100)
n_procs = 4
q_len = 4

def get_shape(batch):
  if isinstance(batch,np.ndarray):
    return batch.shape
  else:
    return [get_shape(b) for b in batch]

def get_random():
  while True:
    np.random.seed(int((time.time()%1)*1e6))
    #return np.random.randn(test_size + int(np.random.randn()*10))
    #return [np.random.randn(test_size) for i in range(3)]
    yield [[np.random.randn(*[t + int(np.random.randn()*10) for t in test_size]) for i in range(3)] for j in range(2)]

def decompose(batch):
  if isinstance(batch,np.ndarray): # Leaf level
    return batch.tobytes(), (batch.shape,batch.dtype)
  else:
    byte_arrs, sigs = zip(*[decompose(b) for b in batch])
    return reduce(lambda x,y: x+y, byte_arrs), sigs

def recompose(buf,signature,start=None):
  if isinstance(signature[0][0],int): # Leaf level, [0] gets to shape, [0][0] gets to first entry in shape
    shape,dtype = signature
    return np.frombuffer(buf,dtype=dtype,offset=start,count=np.prod(shape)).reshape(shape)
  else:
    parts_lens = [get_sig_lengths(s) for s in signature]
    cum_lens = [0] + np.cumsum(parts_lens).tolist()
    return [recompose(buf,sig_i,start=start) for start,sig_i in zip(cum_lens[0:-1],signature)]

def get_sig_lengths(s):
  if isinstance(s[0][0],int): # Leaf level, [0] gets to shape, [0][0] gets to first entry in shape
    shape,dtype = s
    return int(np.prod(shape))*dtype.itemsize
  else:
    return sum(get_sig_lengths(s_i) for s_i in s)

class FastSHMQueue(object):
  def __init__(self,sampler,q_len=8,n_procs=8,debug=False):
    self.sampler = sampler
    self.done = False
    self.debug = debug
  
  def start_init(self,q_len,n_procs):
    self.manager = Manager()
    self.result_queue = self.manager.Queue(q_len)
    self.used_shm_ind_queue = self.manager.Queue(q_len)
    self.bufs = [None]*q_len
    for i in range(q_len):
      self.used_shm_ind_queue.put(i)
    
    self.n_procs = n_procs
    self._last_buf_ind = None
    self._max_buf_size = 0

  def start(self,workers=4,max_queue_size=4):
    self.start_init(max_queue_size,workers)
    
    self.threads = []
    for i in range(self.n_procs):
      t = threading.Thread(target=self._proc_manager_thread_run)
      t.daemon = True
      t.start()
      self.threads.append(t)

  def stop(self,timeout=None):
    self.done = True
    
    # Fill pulling queue so nobody blocks on get
    for i in range(self.n_procs):
      self.used_shm_ind_queue.put(None)
    
    # Empty pushing queue so nobody blocks on put
    while not self.result_queue.empty():
      self.result_queue.get(block=True)
      self.result_queue.task_done()
    
    for t in self.threads:
      t.join(timeout)

  def _proc_manager_thread_run(self):
    while not self.done:
      buf_ind = self.used_shm_ind_queue.get(block=True)
      if buf_ind is None:
        # Done
        return
      buf = self.bufs[buf_ind]
      p = Process(target=self._put_batch,args=(buf_ind,buf,self.result_queue))
      p.daemon = True
      p.start()
      p.join()

  def _put_batch(self,buf_ind,buf,result_queue):
    batch = next(self.sampler)
    
    batch_serial, batch_signatures = decompose(batch)
    
    if buf is None or len(batch_serial) > len(buf):
      # If the buffer is too small or doesn't exist yet we make a new one
      # Use bytes instead of true dtype so batches can have heterogenous dtypes
      del buf
      result_queue.put((batch_serial, batch_signatures, buf_ind))
    else:
      buf[:len(batch_serial)] = batch_serial
      result_queue.put((None, batch_signatures, buf_ind))

  def get(self):
    if not self._last_buf_ind is None:
      self.used_shm_ind_queue.put(self._last_buf_ind)
    buf, batch_signature, buf_ind = self.result_queue.get(block=True)
    self.result_queue.task_done()
    
    if buf is None: # In SHM buf
      if self.debug: print("Get from SHM")
      buf = self.bufs[buf_ind].get_obj()
      self._max_buf_size = max(len(buf),self._max_buf_size)
    else: 
      if self.debug: print("Get from queue")
      self._max_buf_size = max(len(buf),self._max_buf_size)
      self.bufs[buf_ind] = Array('b',self._max_buf_size)
    
    self._last_buf_ind = buf_ind
    return recompose(buf, batch_signature)

if __name__ == "__main__":
  dbl_q = FastSHMQueue(get_random(),debug=True)
  dbl_q.start()
  
  for i in range(20):
    time.sleep(3) # Doing something else...

    tic = time.time()
    batch = dbl_q.get()
    #print(f"{batch[:5]}...({batch.shape}), {time.time()-tic} s")
    print(f"{get_shape(batch)}, {time.time()-tic} s")
    #print(f"{time.time()-t0} seconds")

  print("Stopping...")
  dbl_q.stop()
