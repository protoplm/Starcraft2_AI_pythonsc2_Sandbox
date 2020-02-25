import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import cv2
import keras

import random
import math
import time

from itertools import islice

import sc2
from sc2 import UnitTypeId, AbilityId
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.constants import *
from sc2.player import Bot, Computer
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2, Point3
from creep_manager import CreepManager


import random, json, time
from collections import OrderedDict

# download maps from https://github.com/Blizzard/s2client-proto#map-packs

from sc2.data import race_gas, race_worker, race_townhalls, ActionResult, Attribute, Race

from sc2.position import Point2

from sc2.bot_ai import BotAI
from terrain import Terrain
bot = sc2.BotAI
ter = Terrain(bot)
crp = CreepManager(bot,ter)

class ZergRushBot(sc2.BotAI):

    def __init__(self, use_model=False, title=1):
        #self.MAX_WORKERS = 50
        self.do_something_after = 0
        
        self.title = title
        # DICT {UNIT_ID:LOCATION}
        # every iteration, make sure that unit id still exists!
        self.scouts_and_spots = {}
        self.train_data_queens = []
        self.train_data_larva = []
        self.train_data_attack = []
        self.train_data_overlord = []
        self.train_data_tech = []

        self.attack_model = keras.models.load_model("testing_attack_train1")


        self.fogged_units = []
        self.better_fogged_units = []
        self.fogged_buildings = []
        self.fogged_queue = []

        self.creeploc = []
        self.tumors = []
        self.creep_queen_counter = 0
        self.creep_queens = []

        self.target_grid = []
        self.expansion_p2 = []

        self.lingspeed = 0
        self.roachspeed = 0


        self.use_model = use_model
        if self.use_model:
            print("using model!")
            self.model = keras.models.load_model("BasicCNN-5000-epochs-0.001-LR-STAGE3")

    async def on_start(self):
        self.terrain: Terrain = Terrain(self)
        self.creep_manager: CreepManager = CreepManager(self, self.terrain)
        
        await self.terrain.store_map_feature_coordinates()
        await self.terrain.calculate_expansion_path_distances()
        await self.map_chunks()
        
            
    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result)
        if game_result == Result.Victory:
            with open("wl.txt", "a") as myfile:
                myfile.write("1\n")
            np.save("train_data_queens/{}.npy".format(str(int(time.time()))), np.array(self.train_data_queens))
            np.save("train_data_larva/{}.npy".format(str(int(time.time()))), np.array(self.train_data_larva))
            np.save("train_data_attack/{}.npy".format(str(int(time.time()))), np.array(self.train_data_attack))
            np.save("train_data_overlord/{}.npy".format(str(int(time.time()))), np.array(self.train_data_overlord))
            np.save("train_data_tech/{}.npy".format(str(int(time.time()))), np.array(self.train_data_tech))
        else:
            with open("wl.txt", "a") as myfile:
                myfile.write("0\n")                                  
        self.iteration = 0
        done = 1

    async def on_step(self, iteration):
        self.iteration = iteration
       
        
        

        if iteration == 0:
            done = 0
            await self.chat_send("(glhf)")
            for key in self.expansion_locations:
                self.expansion_p2.append(key)
        # 1000 = ~1:30
        if iteration > 30:
            await self.Clean_vision()
            if iteration % 10 == 0:  
                await self.auto_inject()
            
            if iteration % 50 == 0:
                
                await self.intel()
                await self.do_larva()
            if iteration % 100 == 0:
                await self.distribute_workers()
                await self.Panic()
                await self.do_queens()
                await self.do_attack()
                await self.do_overlords()
                await self.do_tech()
            if iteration % 150 == 0 and iteration < 2000:
                await self.idleamove()
                print((self.time)/60)
            #if iteration > 2500:
                #if iteration % 20:
            if iteration % 20 == 0 and iteration > 100 and iteration < 2000:
                if len(self.creep_queens) > 0:
                    self.terrain.update_creep_coordinates()
                    await self.creep_manager.spread_creep(self.creep_queens) #Spiny's creep
            if iteration > 2000 and iteration % 100 == 0:
                if len(self.creep_queens) > 0:
                    self.terrain.update_creep_coordinates()
                    await self.creep_manager.spread_creep(self.creep_queens) #Spiny's creep
            if iteration % 150 == 0 and iteration > 3500:
                await self.idleamove_end()
                    
                


    async def on_enemy_unit_left_vision(self, unit_tag: int):
        last_known_unit = self._enemy_units_previous_map.get(unit_tag, None) 
        self.fogged_queue.append(last_known_unit.position)
        if len(self.fogged_queue) > 7:
            self.fogged_units.append(Point2.center(self.fogged_queue))
            self.fogged_queue = []

    

    async def Clean_vision(self):
        for unit in self.fogged_units:
            pos = unit.position
            if self.is_visible(pos):
                self.fogged_units.remove(unit)
                            

    async def intel(self):
        game_data = np.zeros((200, 200, 3), np.uint8)


        for unit in self.fogged_units:
            cv2.circle(game_data, (int(unit[0]), int(unit[1])), int(2*2), (116, 116, 116), math.ceil(int(2*0.5)))


        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*5), (255, 255, 255), math.ceil(int(unit.radius*0.5)))

        for unit in self.enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*5), (125, 125, 125), math.ceil(int(unit.radius*0.5)))
        for unit in self.enemy_structures:
            if unit.is_snapshot:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*5), (114, 114, 114), math.ceil(int(unit.radius*0.5)))
            else:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*5), (125, 125, 125), math.ceil(int(unit.radius*0.5)))
        try:
            line_max = 50
            mineral_ratio = self.minerals / 1500
            if mineral_ratio > 1.0:
                mineral_ratio = 1.0

            vespene_ratio = self.vespene / 1500
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0

            population_ratio = self.supply_left / self.supply_cap
            if population_ratio > 1.0:
                population_ratio = 1.0

            plausible_supply = self.supply_cap / 200.0

            worker_weight = len(self.units(UnitTypeId.DRONE)) / (self.supply_cap-self.supply_left)
            if worker_weight > 1.0:
                worker_weight = 1.0

            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

            
        except Exception as e:
            print(str(e))


        # flip horizontally to make our final fix in visual representation:
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)

        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        cv2.imshow(str(self.title), resized)
        cv2.waitKey(1)
        

    async def Expand(self):
        await self.chat_send("Expand")
        await self.expand_now()

    async def Scout(self):
        await self.chat_send("Scout")
        if len(self.units(UnitTypeId.DRONE)) > 0:
            pos = (self.enemy_start_locations[0]).position
            scout = self.workers.closest_to(pos)
            self.do(scout.move(pos))

    
    ############################################################################################
    ######### Queens
    #*********************************************************************************************

    async def do_queens(self):
        if self.structures(UnitTypeId.HATCHERY).ready.amount > 0 and self.can_afford(UnitTypeId.QUEEN):
            if self.use_model:
                prediction = self.queen_model.predict([self.flipped.reshape([-1,200,200,1])])
                choice = np.argmax(prediction[0])
            else:
                choice = random.randrange(0, 3)
                if choice == 0:
                    print("no queen")
                elif choice == 1:
                    if self.structures(UnitTypeId.SPAWNINGPOOL).ready.amount > 0:
                        await self.train_creep_queen()
                    else:
                         choice = None
                elif choice == 2:
                    if self.structures(UnitTypeId.SPAWNINGPOOL).ready.amount > 0:
                        await self.train_inject_queen()
                    else:
                         choice = None
                if choice:
                    y = np.zeros(3)
                    y[choice] = 1
                   
                    self.train_data_queens.append([y,self.flipped])


    async def train_creep_queen(self):
        await self.chat_send("Creep queen")
        if self.can_afford(UnitTypeId.QUEEN):
            self.train(UnitTypeId.QUEEN)
            self.creep_queen_counter += 1

    async def train_inject_queen(self):
        await self.chat_send("Inject queen")
        if self.can_afford(UnitTypeId.QUEEN):
            self.train(UnitTypeId.QUEEN)

    async def auto_inject(self):
        idle_queens = self.units(UnitTypeId.QUEEN).idle.amount
        if idle_queens > 0:
            for hatch in self.townhalls:
                if not hatch.has_buff(BuffId.QUEENSPAWNLARVATIMER):
                    queens = self.units(UnitTypeId.QUEEN).tags_not_in(self.creep_queens)
                    queens = queens.idle
                    if queens:
                        queen = queens.idle.closest_to(hatch.position)
                        if queen.energy >= 25:
                            self.do(queen(AbilityId.EFFECT_INJECTLARVA, hatch)) 

    async def on_unit_created(self, unit: Unit):
        if unit in self.units(UnitTypeId.QUEEN):
            for i in range (self.creep_queen_counter):
                if unit.tag not in self.creep_queens:
                    self.creep_queens.append(unit.tag)
                    self.creep_queen_counter -= 1
       
    #*********************************************************************************************
    ######### Queens
        ############################################################################################
        ############################################################################################
    ######### Map Chopping (Attacks)
    #*********************************************************************************************

    async def do_attack(self):
        attacked = 0
        if self.attack_model:
            prediction = self.attack_model.predict([self.flipped.reshape([-1,200,200,1])])
            randomweight = random.randrange(1,11)
            if randomweight > 2:
                choice = np.argmax(prediction[0])
                await self.chat_send("model attack")
            else:
                choice = random.randrange(0, 10)
                await self.chat_send("randomized attack")
        else:
            
            choice = random.randrange(0, 10)
        if choice == 9:
            attacked = 1
        else:
            if self.units.exclude_type({UnitTypeId.EGG, UnitTypeId.LARVA, UnitTypeId.DRONE, UnitTypeId.OVERLORD, UnitTypeId.QUEEN}).amount > 3:
                await self.chat_send(str(choice) + " attacking")
                await self.amove(choice)
                attacked = 1
        if attacked == 1:
            y = np.zeros(10)
            y[choice] = 1
            print(y)
            self.train_data_attack.append([y,self.flipped])
            

    async def amove(self,choice):
        target = Point2(self.target_grid[choice])
        for unit in self.units.exclude_type({UnitTypeId.EGG, UnitTypeId.LARVA, UnitTypeId.DRONE, UnitTypeId.OVERLORD, UnitTypeId.QUEEN}).idle:
            self.do(unit.attack(target))

    async def idleamove(self):  
        await self.chat_send("idleamove")
        for unit in self.units.exclude_type({UnitTypeId.EGG, UnitTypeId.LARVA, UnitTypeId.DRONE, UnitTypeId.OVERLORD, UnitTypeId.QUEEN}).idle:
            target = unit.position.sort_by_distance(self.expansion_p2)
            self.do(unit.attack(target[0]))    


    async def idleamove_end(self):
        enemy_list = []
        await self.chat_send("idleamove_end")
        for struc in self.enemy_structures:
            enemy_list.append(struc.position)
        if len(enemy_list) > 0:
            for unit in self.units.exclude_type({UnitTypeId.EGG, UnitTypeId.LARVA, UnitTypeId.DRONE, UnitTypeId.OVERLORD, UnitTypeId.QUEEN}).idle:
                target = unit.position.sort_by_distance(enemy_list)
                self.do(unit.attack(target[0]))  
        

    async def map_chunks(self):
        x = self.game_info.map_size[1]
        y = self.game_info.map_size[0]

        chunks = round(math.sqrt(9))

        target_gridx = []
        target_gridy = []
        self.target_grid = []
        u = 0
        for i in range(chunks):
            target_gridy.append(round((y * u) + y/(chunks*2) ))
            u += (1/(chunks))
        u = 0
        for i in range(chunks):
            target_gridx.append(round((x * u) + x/(chunks*2) ))
            u += (1/(chunks))

        self.target_grid = [(x,y) for x in target_gridx for y in target_gridy]


        

   
    #*********************************************************************************************
    ######### Map Chopping (Attacks)
    ############################################################################################               
    ############################################################################################
    ######### Supply Management
    #*********************************************************************************************
    async def do_overlords(self):
        if self.use_model:
            prediction = self.overlord_model.predict([self.flipped.reshape([-1,200,200,1])])
            choice = np.argmax(prediction[0])
        else:
            x = self.units(UnitTypeId.LARVA).idle.amount
            if x:
                if x > 3:
                    x = 4
                choice = random.randrange(0, (x))
                if choice == 0:
                    print("no supply")
                else:
                    await self.overlord(choice)
                await self.chat_send("Supply" + str(choice))
                y = np.zeros(4)
                y[choice] = 1
                self.train_data_overlord.append([y,self.flipped])


    async def overlord(self,choice):
        if self.can_afford(UnitTypeId.OVERLORD) and self.units(UnitTypeId.LARVA).idle.amount > 0:
            self.train(UnitTypeId.OVERLORD,choice)
    

    #*********************************************************************************************
    ######### Supply Management
    ############################################################################################               
    ############################################################################################
    ######### Larva Management
    #*********************************************************************************************
    async def do_larva(self):
        

        if self.use_model:
            prediction = self.larva_model.predict([self.flipped.reshape([-1,200,200,1])])
            choice = np.argmax(prediction[0])
        else:
            choice = random.randrange(0,6)

            if choice == 0:
                print("no larva")
            elif choice == 1:
                await self.all_drone()
            elif choice == 2:
                if self.structures(UnitTypeId.SPAWNINGPOOL).ready.amount > 0 :
                    await self.all_ling()
                else:
                    choice = None
            elif choice == 3:
                if self.structures(UnitTypeId.SPAWNINGPOOL).ready.amount > 0 :
                    await self.half_lingdrone()
                else:
                    choice = None
            elif choice == 4:
                if self.structures(UnitTypeId.SPAWNINGPOOL).ready.amount > 0 and \
                   self.structures(UnitTypeId.ROACHWARREN).ready.amount > 0 :
                    await self.half_roachling()
                else:
                    choice = None
            elif choice == 5:
                if self.structures(UnitTypeId.ROACHWARREN).ready.amount > 0 :
                    await self.all_roach()
                else:
                    choice = None
        if choice:
            y = np.zeros(6)
            y[choice] = 1
            
            self.train_data_larva.append([y,self.flipped])

            

    async def all_ling(self):
        await self.chat_send("Ling")
        if self.can_afford(UnitTypeId.ZERGLING) and \
           self.units(UnitTypeId.LARVA).idle.amount > 0 :
            self.train(UnitTypeId.ZERGLING, self.units(UnitTypeId.LARVA).idle.amount)

    async def all_drone(self):
        await self.chat_send("Drone")
        if self.can_afford(UnitTypeId.DRONE) and \
           self.units(UnitTypeId.LARVA).idle.amount > 0:
            self.train(UnitTypeId.DRONE, self.units(UnitTypeId.LARVA).idle.amount)

    async def half_lingdrone(self):
        
        await self.chat_send("LingDrone")
        if self.can_afford(UnitTypeId.DRONE) and \
           self.units(UnitTypeId.LARVA).idle.amount > 0 :
            x = round((self.units(UnitTypeId.LARVA).idle.amount)/2)
            self.train(UnitTypeId.ZERGLING, x)
            self.train(UnitTypeId.DRONE, self.units(UnitTypeId.LARVA).idle.amount - x)
            

    async def half_roachling(self):
        await self.chat_send("RoachLing")
        if self.can_afford(UnitTypeId.ROACH) and self.units(UnitTypeId.LARVA).idle.amount > 0 :                                                                 
            x = round((self.units(UnitTypeId.LARVA).idle.amount)/2)
            self.train(UnitTypeId.ROACH, x)
            self.train(UnitTypeId.ZERGLING, self.units(UnitTypeId.LARVA).idle.amount -x)

    async def all_roach(self):
        await self.chat_send("Roach")
        if self.can_afford(UnitTypeId.ROACH) and \
           self.units(UnitTypeId.LARVA).idle.amount > 0 :
            self.train(UnitTypeId.ROACH, self.units(UnitTypeId.LARVA).idle.amount)

    #*********************************************************************************************
    ######### Larva Management
    ############################################################################################               
    ############################################################################################
    ######### Tech
    #*********************************************************************************************
    async def do_tech(self):
        if self.use_model:
            prediction = self.tech_model.predict([self.flipped.reshape([-1,200,200,1])])
            choice = np.argmax(prediction[0])
        else:
            choice = random.randrange(0,4)

            if choice == 0:
                print("no tech")
            elif choice == 1:
                await self.progress_tech()
            elif choice == 2:
                await self.take_gas()
            elif choice == 3:
                await self.expand_now()

            y = np.zeros(4)
            y[choice] = 1
            #print(y)
            self.train_data_tech.append([y,self.flipped])


    async def take_gas(self):
        if self.can_afford(UnitTypeId.EXTRACTOR):
                if self.workers.amount > 0:
                    drone = self.workers.random
                    target = self.vespene_geyser.closest_to(drone.position)
                    err = self.do(drone.build(UnitTypeId.EXTRACTOR, target))

    
    async def progress_tech(self):
        await self.chat_send("Tech")
        if self.structures(UnitTypeId.SPAWNINGPOOL).amount + self.already_pending(UnitTypeId.SPAWNINGPOOL) == 0:
            if self.can_afford(UnitTypeId.SPAWNINGPOOL):
                for d in range(5, 6):
                    hatch: Unit = self.townhalls[0]
                    pos = hatch.position.towards(self.game_info.map_center, d)
                    drone = self.workers.closest_to(pos)
                    self.do (drone.build(UnitTypeId.SPAWNINGPOOL, pos)) 
        elif self.structures(UnitTypeId.SPAWNINGPOOL).amount == 1 and self.lingspeed == 0 and self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED):
            self.research(UpgradeId.ZERGLINGMOVEMENTSPEED)
            self.lingspeed = 1
        elif self.structures(UnitTypeId.ROACHWARREN).amount + self.already_pending(UnitTypeId.ROACHWARREN) == 0:
            if self.can_afford(UnitTypeId.ROACHWARREN):
                for d in range(4, 6):
                    hatch: Unit = self.townhalls[0]
                    pos = hatch.position.towards(self.game_info.map_center, d)
                    drone = self.workers.closest_to(pos)
                    self.do (drone.build(UnitTypeId.ROACHWARREN, pos)) 
        elif self.structures(UnitTypeId.LAIR).amount + self.already_pending(UnitTypeId.LAIR) == 0:
                        for hatch in self.townhalls:
                            if hatch.is_idle:
                                self.do(hatch.build(UnitTypeId.LAIR))
                                break
        elif self.structures(UnitTypeId.LAIR).amount == 1 and self.structures(UnitTypeId.ROACHWARREN).amount == 1 \
         and self.roachspeed == 0 and self.can_afford(UpgradeId.GLIALRECONSTITUTION) and self.already_pending_upgrade(UpgradeId.GLIALRECONSTITUTION) == 0:
                        for unit in self.structures(UnitTypeId.ROACHWARREN):
                            self.do(unit(AbilityId.RESEARCH_GLIALREGENERATION))
                            self.roachspeed = 1

    #*********************************************************************************************
    ######### Tech
    ############################################################################################
    async def Panic(self):
        if not self.townhalls:
            for unit in self.units.exclude_type({UnitTypeId.EGG, UnitTypeId.LARVA}):
                await self.chat_send("oshit")
                self.do(unit.attack(self.enemy_start_locations[0]))
            return


    async def get_circle(self,unit):
            loc = unit.position.to2
            positions = [Point2(( \
                int(loc.x + distance * math.cos(math.pi * 2 * alpha / 12)), \
                int(loc.y + distance * math.sin(math.pi * 2 * alpha / 12)))) \
                for alpha in range(12) # alpha is the angle here, locationAmount is the variable on how accurate the attempts look like a circle (= how many points on a circle)
                for distance in range(9, 10)] # distance depending on minrange and maxrange
            return positions
    

    async def draw_creep_pixelmap(self):
        self.creep_map = []
        for (y, x), value in np.ndenumerate(self.state.creep.data_numpy):
            p = Point2((x, y))
            pos = Point2((p.x, p.y))
            if value == 1:
                self.creep_map.append(pos)
                

def main():
    sc2.run_game(
        sc2.maps.get("BelShirVestigeLE"),
        [Bot(Race.Zerg, ZergRushBot(use_model=False)), Computer(Race.Terran, Difficulty.Medium)],
        realtime=False,
    )


main()


