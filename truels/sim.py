from __future__ import annotations
from dataclasses import dataclass , field
import logging
import random
from typing import ClassVar
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import abstractmethod
import concurrent.futures
import math

logging.basicConfig(level = logging.ERROR ,
                    filename="combat.log" ,
                    format = "%(message)s",
                    )

@dataclass
class PROC:

    proc_chance : float
    stat : str
    stat_boost : int
    actions_duration : int
    ticks : int = field(init = False , default = 0)
    is_active : bool = False
    owner : Player | None = field(init = False , default = None , repr = False)

    def check_conditions(self , die_hit_success : bool ) -> None:
        assert isinstance(self.owner , Player)
        self.ticks += 1
        die_roll_success = random.random() <= self.proc_chance if die_hit_success else False
        if self.is_active:
            if die_roll_success:
                ##refreshes the effect if its already activate
                self.ticks = 0

            if self.ticks > self.actions_duration:
                self.deactivate()
        else:
            if die_roll_success:
                self.activate()

    def activate(self) -> None:
        setattr(self.owner , self.stat , getattr(self.owner , self.stat) + self.stat_boost)
        self.is_active = True
        self.ticks = 0

    def deactivate(self) -> None:
        setattr(self.owner , self.stat , getattr(self.owner , self.stat) - self.stat_boost)
        self.is_active = False
        self.ticks = 0

all_procs = {"Crusaders" : PROC(proc_chance = 0.1 ,
                                stat = "damage" ,
                                stat_boost = 10 ,
                                rounds_duration = 2,)}

@dataclass(slots=True)
class Player:

    name : str
    health : int
    damage : int
    armor : int
    hit_chance : float
    _crit_chance : float
    proc_effects : list[PROC] = field(init = False , default_factory = list , repr = False)
    max_health : int = field(init = False , repr = False)
    target : Player | None = field(init = False , repr = False , default = None)
    players : ClassVar[list[Player]] = []
    combat_events : ClassVar[list[str]] = ["miss","crit","normal"]

    def __post_init__(self) -> None:
        ##Testing attribute types and values
        assert isinstance(self._crit_chance , float) and 1>= self._crit_chance >=0.0
        assert isinstance(self.armor , int)
        assert isinstance(self.health , int) and self.health > 0
        assert isinstance(self.damage,  int) and self.damage > 0
        assert isinstance(self.hit_chance , float) and 1>= self.hit_chance >= 0.0

        Player.players.append(self)
        self.max_health = self.health
        logging.info(f"Player {self.name} just landed!")

    def add_proc(self , proc_name : str) -> None:
        proc_effect = all_procs[proc_name]
        proc_effect.owner = self
        self.proc_effects.append(proc_effect)

    @staticmethod
    def damage_reduction(armor : int , cap : float = 1) -> float:
        return 1.0 - min(cap , armor/(armor + 100)) if armor >=0 else 1.0

    @staticmethod
    def get_alive_players() -> list[Player]:
        return [player for player in Player.players if player.is_alive]

    def get_opponents(self) -> list[Player]:
        return [player for player in Player.get_alive_players() if player is not self]

    def revive(self) -> None:
        self.health = self.max_health

    @staticmethod
    def revive_all() -> None:
        for p in Player.players:
            p.revive()

    @classmethod
    def get_player(cls, name : str , hit_chance : float) -> Player:
        return cls(name=name,hit_chance=hit_chance,damage=10,health=1)

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    @abstractmethod
    def set_target(self) -> Player:
       """Fight Protocol"""
       pass

    @property
    def reduced_damage(self) -> int:
        return math.ceil(self.damage*Player.damage_reduction(self.target.armor))

    @property
    def miss_chance(self) -> float:
        return 1 - self.hit_chance

    @property
    def crit_chance(self) -> float:
        return min( 1 - self.miss_chance , self._crit_chance)

    def get_attack_table(self) -> list[float]:
        return [self.miss_chance , self.crit_chance , 1 - self.miss_chance - self.crit_chance]

    def get_combat_event(self) -> str:
        """Returns the combat event"""
        attack_table = self.get_attack_table()
        for i in Player.combat_events:
            if random.random() <= sum(attack_table[:i+1]):
                return i


    def attack(self) -> None:
        if self.target:
            combat_event = self.get_combat_event()

            if combat_event != "miss":
                is_critical = int(combat_event == "crit")
                additional_message = " (crit) " if is_critical else ""
                self.target.health -= self.reduced_damage*(2 - int(combat_event == "crit"))
                logging.info(f"Player {self.name} strikes {self.target.name} for {self.damage}{additional_message}!")

            for proc in self.proc_effects:
                proc.check_conditions(combat_event != "miss")


class RandomFiringBot(Player):

    def set_target(self) -> Player:
        """Randomly Fires at a Target"""
        self.target = random.choice(self.get_opponents())
        return self.target

class MaxHitChanceBot(Player):

    def set_target(self) -> Player:
        """Always hits the strongest Opponent!"""
        self.target = max(self.get_opponents() , key = lambda p : p.hit_chance)
        logging.info(f"{self.name} stares aggresively at {self.target.name}!")
        return self.target

from typing import Iterable

def is_valid_distr(*p , err : float = 0.01) -> bool:
    return abs(sum(p)-1) < err and all(1.0 > p0 > 0.0 for p0 in p)

def main():
    from collections import defaultdict
    from itertools import product
    import numpy as np
    import pandas as pd
    import plotly.express as px

    complete_history = defaultdict(lambda : defaultdict(int))
    ##Simultaneous truels can end with 0 or 1 survivor

    ##monte-carlo init
    MAX_TRIALS = 5000

    ##probability configuration
    prob_config = [np.arange(0.1 , 1.1 , 0.1)]*2
    df = pd.DataFrame(columns = ["p_alice","p_bob","p_nicole","alice_out","bob_out","nicole_out","draw_out"])
    outcomes = ["alice_out","bob_out","nicole_out","draw_out"]

    combinations = list(product(*prob_config))
    for idx , p_config in tqdm(enumerate(combinations)):

        p_alice , p_bob = p_config
        p_nicole = max(0, 1 - p_alice - p_bob)

        if not is_valid_distr(p_alice , p_bob , p_nicole):
            print("Skipping...")
            continue

        if idx > 10:
            break

        print(p_alice , p_bob , p_nicole)

        history = complete_history[(p_alice , p_bob , p_nicole)]

        ##create new players
        players = [MaxHitChanceBot.get_player("Alice",p_alice),
                   MaxHitChanceBot.get_player("Bob",p_bob),
                   MaxHitChanceBot.get_player("Nicole",p_nicole)
                  ]
        for _ in tqdm(range(MAX_TRIALS) , leave = False):

            Player.revive_all()
            rnd = 1
            while True:
                logging.info(f"Round {rnd} starts!")
                alive_players = Player.get_alive_players()
                if len(alive_players) < 2:
                    logging.info("Game complete")
                    break

                ##setting targets
                for p in alive_players:
                    p.set_target()

                for p in alive_players:
                    p.attack()

                for p in alive_players:
                    if not p.is_alive:
                        logging.info(f"On round {rnd} {p.name} has deceased!")

                rnd += 1

            ##append results after simulation finishes
            match alive_players:
                case []:
                    history["Draw"] += 1
                case [*args]:
                    history[args[0].name] += 1
                case _:

                    class UnknownPlayer(Exception):
                        pass
                    raise UnknownPlayer("Something Went Wrong")

            logging.info("Game over!")
            logging.info(f"Winners: {alive_players}!")

        df.loc[idx , ["p_alice","p_bob","p_nicole"]] = p_alice , p_bob , p_nicole
        ##normalizing to get winning probabilities
        for i , (key , value) in enumerate(history.items()):
            history[key] = value/float(MAX_TRIALS)

            df.loc[idx , outcomes[i]] = history[key]

        logging.info("Simulation complete!")

        #visualize a simulation for a particular probability
        #plt.bar(history.keys() , history.values())
        #plt.show()

    ##visualize results for varying probabilities
    #plt.style.use("dark_background")
    #plt.plot(complete_history.keys() , [history["Nicole"] for history in complete_history.values()] ,
    #                                    linestyle = '--',
    #                                    color = 'lightyellow'
    #                                    )
    #plt.show()



    fig = px.scatter_ternary(df , a = "p_alice" , b = "p_bob" , c = "p_nicole" , size = "nicole_out")
    fig.show()


if __name__ == "__main__":
    main()
