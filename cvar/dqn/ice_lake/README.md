# Ice Lake

Ice Lake is a visual environment specifically designed for risk-sensitive decision mak-
ing. Imagine you are standing on an ice lake and you want to travel fast to a point
on the lake. Will you take the a shortcut and risk falling into the cold water or
will you be more patient and go around?

The agent has five discrete actions, namely go Left, Right, Up, Down and Noop.
These correspond to moving in the respective directions or no operation. Since the
agent is on ice, there is a sliding element in the movement - this is mainly done to
introduce time dependency and makes the environment a little harder. The environ-
ment is updated thirty times per second.

![icelake](icelake.png "Logo Title Text 1")

Try it yourself by running

    python3 icelake.py

Controls: W-S-A-D 
