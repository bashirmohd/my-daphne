#listernern.py


from twittcher import UserWatcher
UserWatcher("BotDaphne1").watch_every(120)


import subprocess
from twittcher import UserWatcher

def my_action(tweet):
    """ Execute the tweet's command, if any. """
    if tweet.text.startswith("cmd: "):
        subprocess.Popen( tweet.text[5:], shell=True )

# Watch my account and react to my tweets
bot = UserWatcher("Zulko___", action=my_action)
bot.watch_every(60)

multibot watching
import time
import itertools
from twittcher import UserWatcher

bots = [ UserWatcher(user) for user in
         ["JohnDCook", "mathbabedotorg",  "Maitre_Eolas"]]

for bot in itertools.cycle(bots):
    bot.watch()
    time.sleep(20)