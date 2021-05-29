import tensorflow as tf
from fillmore.utils import WarmupLRScheduler

class LearningRateSchedulerTest(tf.test.TestCase):
    def setUp(self):
        self.scale = 2e-3
        self.hidden_size = 768
        self.total_steps = 600

    def test_warmup_scheduler(self):
        warm_up_prop = 0.1
        lr_scheduler = WarmupLRScheduler(self.hidden_size, warmup_steps=self.total_steps*warm_up_prop)
        lrs = [lr_scheduler(tf.cast(step,tf.float32)) for step in range(self.total_steps)]
        max_lr = tf.reduce_max(lrs).numpy()
        self.assertAllClose(max_lr, 9.31695e-06)