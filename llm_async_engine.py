import multiprocessing as mp

class AsyncLLMEngine:
    def __init__(self, model_path, llm_config, num_workers=1):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.num_workers = num_workers
        self.model_path = model_path
        self.llm_config = llm_config
        self.workers = []
        for i in range(num_workers):
            proc = mp.Process(target=self.worker_loop, args=(self.task_queue, self.result_queue, model_path, llm_config, i))
            proc.daemon = True
            proc.start()
            self.workers.append(proc)
    @staticmethod
    def worker_loop(task_queue, result_queue, model_path, llm_config, worker_id):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from __main__ import CustomLLM  # 确保CustomLLM可见
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            load_in_4bit=True,
        )
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        llm = CustomLLM(model, tokenizer, llm_config)
        while True:
            task = task_queue.get()
            if task is None:
                break
            prompt, task_id = task['prompt'], task['task_id']
            try:
                output, metrics, hidden = llm.invoke_with_metrics(prompt)
                result_queue.put({'task_id': task_id, 'output': output, 'metrics': metrics, 'hidden': hidden})
            except Exception as e:
                result_queue.put({'task_id': task_id, 'error': str(e), 'output': None, 'metrics': {}, 'hidden': (None, None, None)})
    def submit(self, prompt, task_id):
        self.task_queue.put({'prompt': prompt, 'task_id': task_id})
    def get_result(self, timeout=None):
        return self.result_queue.get(timeout=timeout)
    def stop(self):
        for _ in self.workers:
            self.task_queue.put(None)
        for w in self.workers:
            w.join()
