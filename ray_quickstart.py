# ray_quickstart.py
# https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8

# Make sure to use virtual environment (.venv) to keep Python project, isolate project dependencies, and versions
# python3 -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt or pip install package_name (I installed numpy, ray)
# pip freeze > requirements.txt (push to github)
# deactivate

'''
A Simple Python Example: Running a Ray Task on a Remote Cluster

I. Create function f and turn it into a "remote function" - that is a function that can be executed remotely & async'ly

Ray's remote cluster allows you to scale your computations beyond the resources of a single machine, enabling large-scale parallel processing and distributed computing. You can add or remove nodes from the cluster as needed, and Ray will automatically rebalance the workload to make efficient use of the available resources.
'''

import numpy as np
import ray
import time

# initialize ray
ray.init()

# # Define the sleeping task.
@ray.remote
def f(x):
    time.sleep(1)
    return x

# Launch 4 tasks in parallel
result_ids = []
for i in range(4):
    result_ids.append(f.remote(i))

# Wait for the tasks to complete.
results = ray.get(result_ids)

# I was able to get this up and running:
# 2024-01-09 13:47:54,839	INFO worker.py:1715 -- Started a local Ray instance. View the dashboard at 127.0.0.1:8265 

print(results)



'''
II. Task Dependencies
- Tasks can depend on other tasks
- Here, z_id depends on both x_id and y_id completing before it can execute multiply_matrices - uses the outputs of the two tasks
- In Ray, when you call a remote function using the .remote() method, it returns an ID that represents the task being executed. This ID can be used to retrieve the result of the task once it has completed.

- In summary, z = ray.get(z_id) retrieves the result of the matrix multiplication, which was executed remotely, and assigns it to the variable z.

- *Think of it like a mailbox*: you send a letter (the task) and get a mailbox key (the task ID). Later, you use the key to retrieve the letter (the result) from the mailbox. In this example, ray.get(z_id) is like retrieving the letter from the mailbox, and the letter is the multiplied matrix, which is assigned to z.


'''
@ray.remote
def create_matrix(size):
    return np.random.normal(size=size)

@ray.remote
def multiply_matrices(x, y):
    return np.dot(x, y)

x_id = create_matrix.remote([1000, 1000])
y_id = create_matrix.remote([1000, 1000])
z_id = multiply_matrices.remote(x_id, y_id) # line will not execute until the first 2 tasks are completed running

# Get the results.
# ray.get() is a way to wait for the remote function to finish executing. 
# Ray blocks the main thread until the results are available, then returns the result.
z = ray.get(z_id)


''' 

III. Use @ray.remote tag whenever you want a function to move to PARALLEL PROCESSING

Scaling up - Dependency graphs with depth 3 vs 7

In this case, changing a single line of code can change the aggregation’s running time from linear to logarithmic in the number of values being aggregated.

As described above, to feed the output of one task as an input into a subsequent task, simply pass the future returned by the first task as an argument into the second task. 

This task dependency will automatically be taken into account by Ray’s scheduler. The second task will not execute until the first task has finished, and the output of the first task will automatically be shipped to the machine on which the second task is executing.

When you call: add.remote(x, y), Ray will schedule it to be executed on a node in the cluster. The node will execute the function, which will pause for 1 second and then return the result. 

The result will then be returned to the client as a Future object, which can be used to retrieve the result when it's available.

'''

@ray.remote
def add(x, y):
    '''
    This basic sleep and add function is decorated with `@ray.remote`, which indicates that it should be executed on a Ray cluster.
    '''
    time.sleep(1)
    return x + y


# This way aggregates values slowly and takes O(n) time (7 seconds)
id1 = add.remote(1, 2)
id2 = add.remote(id1, 3)
id3 = add.remote(id2, 4)
id4 = add.remote(id3, 5)
id5 = add.remote(id4, 6)
id6 = add.remote(id5, 7)
id7 = add.remote(id6, 8)
result = ray.get(id7)


# This way aggregates values slowly and takes O(log(n)) time (3 seconds)
id1 = add.remote(1, 2)
id2 = add.remote(3, 4)
id3 = add.remote(5, 6)
id4 = add.remote(7, 8)
id5 = add.remote(id1, id2)
id6 = add.remote(id3, id4)
id7 = add.remote(id5, id6)
result = ray.get(id7)

# More concise while loops:
# Slow
# values = [1, 2, 3, 4, 5, 6, 7, 8]
# while len(values) > 1:
#     values = [add.remote(values[0], values[1], values[2:])] # interesting syntax
# result = ray.get(values[0])

# Fast approach.
values = [1, 2, 3, 4, 5, 6, 7, 8]
while len(values) > 1:
    values = values[2:] + [add.remote(values[0], values[1])]
result = ray.get(values[0])



'''
IV. Python classes become ACTORS with the @ray.remote decorator.

Actors allow mutable state to be shared between tasks in a way that remote functions do not.

- Classes are essential in a distributed setting as well on a single core.
- When class is instantiated, ray creas a new actor - a process that runs somewhere in the cluster and holds a copy of the object.
- Method invocations on that actor turn into Tasks.
- Those run on the actor process and can access and mutate the state of the actor.

Individual actors execute methods serially.
- Each method is atomic, no race conditions

* Parallelism can be achieved by creating multiple actors. *

Here is a simplest possible usage example for actors.


'''

@ray.remote
class Counter(object):
    
    def __init__(self):
        self.x = 0
    
    def inc(self):
        self.x += 1
    
    def get_value(self):
        return self.x

# Create an actor process. This has a copy of the Counter object.
    # c = Counter.remote(): This creates a new instance of the Counter class and marks it as a remote actor. 
    # This instance will be managed by the Ray library, allowing it to be executed on a remote machine.

# Note: This is in main...
c = Counter.remote()

# Check the actor's counter value.
print(ray.get(c.get_value.remote()))  # 0

# Increment the counter twice and check the value again.
c.inc.remote()
c.inc.remote()
print(ray.get(c.get_value.remote()))  # 2




'''
V. Actor Handles

Actors are extremely powerful. They allow you to take a Python class and instantiate it as a microservice which can be queried from other actors and tasks and even other applications.

Above, we only invoked methods on the actor from the main python script. But, you can pass around handles to an actor, allowing other actors or tasks to invoke methods on the same actor.

Ex. `MessageActor` - creates an actor that stores messages. Worker tasks push methods to the actor,

`worker` - pushes messages to the actor

`main` - Create a message actor; the Python main script reads the messages periodically.

'''
@ray.remote
class MessageActor(object):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)
    
    def get_and_clear_message(self):
        messages = self.messages
        self.messages = []
        return messages
    

# Define a *remote function* which loops around and pushes messages to the actor.
@ray.remote
def worker(message_actor, j):
    for i in range(100):
        time.sleep(1)
        message_actor.add_message.remote(
            "Message {} from worker {}.".format(i, j))

# Create an instance of message actor.
message_actor = MessageActor.remote()

# start 3 tasks that push messages to the actor.
[worker.remote(message_actor, j) for j in range(3)]

# Get the messages and print them periodically.
for _ in range(100):
    new_messages = ray.get(message_actor.get_and_clear_message.remote())
    print("new messages", new_messages)
    time.sleep(1)







