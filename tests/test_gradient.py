import tensorflow as tf

z = tf.constant(3.0)

x = z


for i in range(2):
  # z = y * y
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x * x
  
  print(tape.gradient(y, x).numpy())
  print(tape.gradient(y, x).numpy())
  x = x + 2.0
# Use the tape to compute the gradient of z with respect to the
# intermediate value y.
# dz_dx = 2 * y, where y = x ** 2


# x = x + 2.0
# print(x)
# print(tape.gradient(y, x))