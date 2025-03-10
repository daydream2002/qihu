#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/7/30 10:49
# @Author  : Joisen
# @File    : test.py.py

import os
import pandas as pd
import json
from feature_extract.feature_extract import *
import numpy as np
from feature_extract.cal_xts import *


def get_dealer(zhuang_id, high_score_id):
    '''
    :param zhuang_id: 庄家id
    :param high_score_id: 高手玩家id
    :return: 进行位置调整后庄家的位置
    '''
    retList = []
    for i in range(4):
        if zhuang_id == i:
            retList.append(1)
        else:
            retList.append(0)
    return changeSeat(retList, high_score_id)


def self_king_num(king_card, handCards0):
    '''
    :param king_card: 宝牌
    :param handCards0: 当前玩家即高手玩家的手牌
    :return: 返回高手玩家的宝牌数
    '''
    ret_num = 0
    for i in range(len(handCards0)):
        if handCards0[i] == king_card:
            ret_num += 1
    return ret_num


def get_feiKing_num(disCardReal, kingCard):
    '''
    :param disCardReal: 经过位置调整后所有玩家真实丢弃的牌
    :param kingCard: 宝牌
    :return: 返回所有玩家的飞宝数  一维list
    '''
    retList = []
    for discard in disCardReal:
        num = 0
        for i in discard:
            if i == kingCard:
                num += 1
        retList.append(num)
    return retList


# 获取牌墙中剩余的牌
def get_remain_card(discards, fulu, hand_cards):
    '''
    :param hand_cards: 各玩家的手牌
    :param discards: 丢弃的牌(不含副露中的牌)
    :param fulu: 副露
    :return: 牌墙中剩余的牌
    '''
    card_num = 0
    for discard in discards:
        card_num += len(discard)
    for fulu_ in fulu:
        for fl in fulu_:
            card_num += len(fl)
    for hand_card in hand_cards:
        card_num += len(hand_card)

    return 136 - card_num


def get_high_score_seatId(player_ids, high_score_id):
    '''
    :param player_ids: 选手id的一维列表
    :param high_score_id: 高分选手的id
    :return: 高分选手的座位号
    '''
    for i in range(len(player_ids)):
        if player_ids[i] == high_score_id:
            return i


def changeSeat(cards, seatId):
    '''
    :param handCards: 所有玩家牌
    :param seatId: 高手玩家的位置
    :return: 返回将高手玩家置为第一位的牌列表
    '''
    ret_list = cards[seatId:] + cards[:seatId]
    return ret_list


def get_chi_position(combine_cards, operate_card):
    '''
    :param combine_cards: 进行吃牌动作后形成的组合牌
    :param operate_card: 吃牌动作的操作牌
    :return: 返回吃牌的位置(左->1、中->2、右->3)
    '''
    for i in range(len(combine_cards)):
        if combine_cards[i] == operate_card:
            return i + 1


def can_eat(handCards0, pre_operate_card) -> bool:
    '''
    :param handCards0: 高手玩家的手牌
    :param pre_operate_card: 上一个玩家丢的牌
    :return: 返回是否能吃但没吃
    操作牌的范围为：1-9，17-25，33-41，49-55
    '''
    # 如果操作牌 为字牌 则返回False
    if pre_operate_card >= 49:
        return False

    # 判断包含操作牌的顺子是否在手牌中
    pre_card_2 = pre_operate_card - 2
    pre_card_1 = pre_operate_card - 1
    rear_1 = pre_operate_card + 1
    rear_2 = pre_operate_card + 2
    if pre_card_2 in handCards0 and pre_card_1 in handCards0:
        return True
    elif pre_card_1 in handCards0 and rear_1 in handCards0:
        return True
    elif rear_1 in handCards0 and rear_2 in handCards0:
        return True

    return False


def can_pong(handCards0, op_card):
    if handCards0.count(op_card) == 2:
        return True
    return False


# 获取丢牌模型需要的信息
def get_json_info(data_dir, store_path):
    # get json file_names
    all_json = os.listdir(data_dir)
    file_num = 79833
    for j_name in all_json:
        # open json file
        j = open(data_dir + j_name, encoding='utf-8')
        # load info in json
        info = json.load(j)
        # print(j_name)
        # 庄家id
        zhuang_id = info['zhuang_id']
        # 得到宝牌
        king_card = info['king_card']
        # 获取高分选手的位置
        high_score_seatId = get_high_score_seatId(info['players_id'], info['high_score_player_id'])

        dealer_flag = get_dealer(zhuang_id, high_score_seatId)

        battle_info = info['battle_info']
        for bat_info in battle_info:
            # 获取battle_info中的丢牌动作的信息
            if bat_info['action_type'] == 'd' and bat_info['seat_id'] == high_score_seatId:
                discards = bat_info['discards']  # 弃牌，不含副露中的牌
                discards_seq = bat_info['discards_real']
                fulu_ = changeSeat(bat_info['discards_op'], high_score_seatId)  # 将高手玩家的副露调整到第一位
                handCards = changeSeat(bat_info['handcards'], high_score_seatId)  # 将高手玩家的牌调整到第一位
                remain_card_num = get_remain_card(discards, fulu_, handCards)
                self_kingNum = self_king_num(king_card, handCards[0])  # 高手玩家的宝牌数
                fei_king_nums = get_feiKing_num(discards_seq, king_card)
                round_ = bat_info['round']
                operate_card = bat_info['operate_card']

                # features = card_preprocess_sr_suphx(handCards[0],fulu_,king_card,discards_seq,remain_card_num,
                #                                     self_kingNum,fei_king_nums,round_,dealer_flag,search=False)
                # print(np.array(features).shape)
                # 将特征存入json文件
                storeFile = store_path + f'{file_num}.json'
                storeDict = {}
                storeDict['handCards0'] = handCards[0]
                storeDict['fulu_'] = fulu_
                storeDict['king_card'] = king_card
                storeDict['discards_seq'] = discards_seq
                storeDict['remain_card_num'] = remain_card_num
                storeDict['self_king_num'] = self_kingNum
                storeDict['fei_king_nums'] = fei_king_nums
                storeDict['discards'] = discards
                storeDict['round_'] = round_
                storeDict['dealer_flag'] = dealer_flag
                storeDict['operate_card'] = operate_card
                with open(storeFile, 'w', encoding="utf-8") as fp:
                    json.dump(storeDict, fp, indent=4)
                    file_num += 1
    print('数量：', file_num)


# 获取吃牌模型需要的信息
# 吃牌模型： 不吃(可以吃但是没有吃)、吃(左吃、中吃、右吃)  分别编码为0、1、2、3
# 不吃时：高手玩家的动作是摸牌，且上一个动作是高手玩家的上家丢牌，丢的牌高手玩家可以吃，但是没有吃。
def get_json_info_chi(data_dir, store_path):
    # get json file_names
    all_json = os.listdir(data_dir)
    file_num_left = 13134
    file_num_mid = 10626
    file_num_right = 13222
    file_num_nochi = 72656

    for j_name in all_json:
        # open json file
        j = open(data_dir + j_name, encoding='utf-8')
        # load info in json
        info = json.load(j)
        # 庄家id
        zhuang_id = info['zhuang_id']
        # 得到宝牌
        king_card = info['king_card']
        # 获取高分选手的位置
        high_score_seatId = get_high_score_seatId(info['players_id'], info['high_score_player_id'])
        # 获得进行位置调整后的庄家位置
        dealer_flag = get_dealer(zhuang_id, high_score_seatId)

        battle_info = info['battle_info']
        # 1、获取高分玩家吃的动作，需要对吃牌是左吃、中吃和右吃进行判断
        # 遍历一个json文件中battle_info的所有动作
        for temp in range(len(battle_info)):
            bat_info = battle_info[temp]  # bat_info -> 当前动作的桌面信息
            # 当temp>=1 时才会发生高手玩家进行吃的动作 或者 可以吃但没有吃的情况（temp初始为0）
            if temp >= 1:
                pre_bat_info = battle_info[temp - 1]  # 当前动作的前一个动作的桌面信息
                # 如果当前动作是高手玩家发生的 且 当前动作是摸牌 且 上一家的动作是丢牌  进行是否能吃但没吃的判断
                if bat_info['action_type'] == 'G' and pre_bat_info['action_type'] == 'd' and bat_info[
                    'seat_id'] == high_score_seatId:
                    # 获取高手玩家的手牌
                    handCards = changeSeat(bat_info['handcards'], high_score_seatId)  # 将高手玩家的牌调整到第一位
                    handCards0 = handCards[0]  # 高手玩家的手牌
                    # 获取当前高手玩家上一个玩家丢的牌
                    pre_operate_card = pre_bat_info['operate_card']
                    # 对高手玩家的手牌和上一个玩家丢的牌进行判断是否高手玩家可以吃
                    if can_eat(handCards0, pre_operate_card):  # 如果为True则提取该动作的所有桌面信息
                        discards = bat_info['discards']  # 弃牌，不含副露中的牌
                        discards_seq = bat_info['discards_real']
                        fulu_ = changeSeat(bat_info['discards_op'], high_score_seatId)  # 将高手玩家的副露调整到第一位
                        remain_card_num = get_remain_card(discards, fulu_, handCards)
                        # print('剩余牌数', remain_card_num)
                        self_kingNum = self_king_num(king_card, handCards0)  # 高手玩家的宝牌数
                        fei_king_nums = get_feiKing_num(discards_seq, king_card)
                        round_ = bat_info['round']
                        # features = card_preprocess_sr_suphx(handCards[0],fulu_,king_card,discards_seq,remain_card_num,
                        #                                     self_kingNum,fei_king_nums,round_,dealer_flag,search=False)
                        # print(np.array(features).shape)
                        # 将特征存入json文件
                        storeFile = store_path + 'no_chi/' + f'{file_num_nochi}_nc.json'
                        storeDict = {}
                        storeDict['handCards0'] = handCards0
                        storeDict['fulu_'] = fulu_
                        storeDict['king_card'] = king_card
                        storeDict['discards_seq'] = discards_seq
                        storeDict['remain_card_num'] = remain_card_num
                        storeDict['self_king_num'] = self_kingNum
                        storeDict['fei_king_nums'] = fei_king_nums
                        storeDict['discards'] = discards
                        storeDict['round_'] = round_
                        storeDict['dealer_flag'] = dealer_flag
                        storeDict['chi_position'] = 0
                        with open(storeFile, 'w', encoding="utf-8") as fp:
                            json.dump(storeDict, fp, indent=4)
                            file_num_nochi += 1
                            # print(j_name)

                # 获取battle_info中的吃牌动作的信息
                elif bat_info['action_type'] == 'C' and bat_info['seat_id'] == high_score_seatId:
                    discards = bat_info['discards']  # 弃牌，不含副露中的牌
                    discards_seq = bat_info['discards_real']
                    fulu_ = changeSeat(bat_info['discards_op'], high_score_seatId)  # 将高手玩家的副露调整到第一位
                    handCards = changeSeat(bat_info['handcards'], high_score_seatId)  # 将高手玩家的牌调整到第一位
                    remain_card_num = get_remain_card(discards, fulu_, handCards)
                    # print('剩余牌数',remain_card_num)
                    self_kingNum = self_king_num(king_card, handCards[0])  # 高手玩家的宝牌数
                    fei_king_nums = get_feiKing_num(discards_seq, king_card)
                    round_ = bat_info['round']
                    operate_card = bat_info['operate_card']

                    # 获得吃牌的位置 左中右->123
                    combine_cards = bat_info['combine_cards']
                    chi_position = get_chi_position(combine_cards, operate_card)
                    # features = card_preprocess_sr_suphx(handCards[0],fulu_,king_card,discards_seq,remain_card_num,
                    #                                     self_kingNum,fei_king_nums,round_,dealer_flag,search=False)
                    # print(np.array(features).shape)
                    # 将特征存入json文件

                    storeDict = {}
                    storeDict['handCards0'] = handCards[0]
                    storeDict['fulu_'] = fulu_
                    storeDict['king_card'] = king_card
                    storeDict['discards_seq'] = discards_seq
                    storeDict['remain_card_num'] = remain_card_num
                    storeDict['self_king_num'] = self_kingNum
                    storeDict['fei_king_nums'] = fei_king_nums
                    storeDict['discards'] = discards
                    storeDict['round_'] = round_
                    storeDict['dealer_flag'] = dealer_flag
                    storeDict['chi_position'] = chi_position
                    if chi_position == 1:
                        storeFile = store_path + 'left/' + f'left{file_num_left}.json'
                        with open(storeFile, 'w', encoding="utf-8") as fp:
                            json.dump(storeDict, fp, indent=4)
                            file_num_left += 1
                            # print(j_name)
                    if chi_position == 2:
                        storeFile = store_path + 'mid/' + f'mid_{file_num_mid}.json'
                        with open(storeFile, 'w', encoding="utf-8") as fp:
                            json.dump(storeDict, fp, indent=4)
                            file_num_mid += 1
                    if chi_position == 3:
                        storeFile = store_path + 'right/' + f'right_{file_num_right}.json'
                        with open(storeFile, 'w', encoding="utf-8") as fp:
                            json.dump(storeDict, fp, indent=4)
                            file_num_right += 1

    print('文件数：左', file_num_left, '->中', file_num_mid, '->右', file_num_right, '->不吃', file_num_nochi)


def get_json_info_pong(data_dir, store_path):
    # get json file_names
    all_json = os.listdir(data_dir)
    file_num_np = 12980
    file_num_p = 68497

    for j_name in all_json:
        # open json file
        j = open(data_dir + j_name, encoding='utf-8')
        # load info in json
        info = json.load(j)
        # 庄家id
        zhuang_id = info['zhuang_id']
        # 得到宝牌
        king_card = info['king_card']
        # 获取高分选手的位置
        high_score_seatId = get_high_score_seatId(info['players_id'], info['high_score_player_id'])
        # 获得进行位置调整后的庄家位置
        dealer_flag = get_dealer(zhuang_id, high_score_seatId)
        battle_info = info['battle_info']
        # 碰：高手玩家的动作是碰。 不碰：当前玩家的动作是丢牌，且在当前玩家发生动作时的手牌信息中高手玩家的手牌可以碰当前玩家丢的牌
        # 获取不碰时信息的步骤：首先获取每一个丢牌动作信息中的手牌信息(即高手玩家的手牌信息 和 当前丢的牌)，对高手玩家是否可以碰进行判断
        # 遍历一个json文件中battle_info的所有动作
        for i in range(len(battle_info)):
            bat_info = battle_info[i]  # 当前动作信息
            # 当前动作是丢牌 且 当前丢牌的不是高手玩家 且 不是最后一个动作
            if bat_info['action_type'] == 'd' and i < len(battle_info) - 1 and bat_info[
                'seat_id'] is not high_score_seatId:  # 当前动作为丢牌
                next_bat_info = battle_info[i + 1]  # 下一个动作信息
                # 获取当前动作的手牌信息
                handCards = changeSeat(bat_info['handcards'], high_score_seatId)  # 将高手玩家的牌调整到第一位
                # 获取高手玩家的手牌：
                handCards0 = handCards[0]
                # 获取当前丢牌动作所丢的牌
                operate_card = bat_info['operate_card']
                # 判断高手玩家当前是否可以碰
                # 如果高手玩家可以碰
                if can_pong(handCards0, operate_card):
                    # 但是下一个动作不是碰，则为不碰
                    if next_bat_info['action_type'] != 'N':
                        # 不碰  将数据写入json文件
                        fulu_ = changeSeat(bat_info['discards_op'], high_score_seatId)
                        discards = bat_info['discards']
                        discards_seq = changeSeat(bat_info['discards_real'], high_score_seatId)
                        remain_card_num = get_remain_card(discards, fulu_, handCards)
                        self_kingNum = self_king_num(king_card, handCards[0])  # 高手玩家的宝牌数
                        fei_king_nums = get_feiKing_num(discards_seq, king_card)
                        round_ = bat_info['round']

                        storeFile = store_path + 'np/' + f'np{file_num_np}.json'
                        storeDict = {}
                        storeDict['handCards0'] = handCards0
                        storeDict['fulu_'] = fulu_
                        storeDict['king_card'] = king_card
                        storeDict['discards_seq'] = discards_seq
                        storeDict['remain_card_num'] = remain_card_num
                        storeDict['self_king_num'] = self_kingNum
                        storeDict['fei_king_nums'] = fei_king_nums
                        storeDict['discards'] = discards
                        storeDict['round_'] = round_
                        storeDict['dealer_flag'] = dealer_flag
                        storeDict['isPong'] = 0  # 不碰为 0
                        with open(storeFile, 'w', encoding="utf-8") as fp:
                            json.dump(storeDict, fp, indent=4)
                            file_num_np += 1

            # 碰牌： 当前动作是 Ｎ　且发生当前动作的是高手玩家
            elif bat_info['action_type'] == 'N' and bat_info['seat_id'] == high_score_seatId:
                handCards = changeSeat(bat_info['handcards'], high_score_seatId)  # 将高手玩家的牌调整到第一位
                handCards0 = handCards[0]
                discards = bat_info['discards']  # 弃牌，不含副露中的牌
                discards_seq = changeSeat(bat_info['discards_real'], high_score_seatId)
                fulu_ = changeSeat(bat_info['discards_op'], high_score_seatId)  # 将高手玩家的副露调整到第一位
                remain_card_num = get_remain_card(discards, fulu_, handCards)
                self_kingNum = self_king_num(king_card, handCards0)  # 高手玩家的宝牌数
                fei_king_nums = get_feiKing_num(discards_seq, king_card)
                round_ = bat_info['round']
                # 将特征存入json文件
                storeFile = store_path + 'p/' + f'p{file_num_p}.json'
                storeDict = {}
                storeDict['handCards0'] = handCards0
                storeDict['fulu_'] = fulu_
                storeDict['king_card'] = king_card
                storeDict['discards_seq'] = discards_seq
                storeDict['remain_card_num'] = remain_card_num
                storeDict['self_king_num'] = self_kingNum
                storeDict['fei_king_nums'] = fei_king_nums
                storeDict['discards'] = discards
                storeDict['round_'] = round_
                storeDict['dealer_flag'] = dealer_flag
                storeDict['isPong'] = 1  # 碰为 1
                with open(storeFile, 'w', encoding="utf-8") as fp:
                    json.dump(storeDict, fp, indent=4)
                    file_num_p += 1

    print('文件数：不碰', file_num_np, '->碰', file_num_p)


def get_json_info_hu(data_dir, store_path):
    global file_num_nh, file_num_h
    # get json file_names
    all_json = os.listdir(data_dir)
    for j_name in all_json:
        j = open(data_dir + j_name, encoding='utf-8')
        info = json.load(j)
        # 庄家id
        zhuang_id = info['zhuang_id']
        # 得到宝牌
        king_card = info['king_card']
        # 获取高分选手的位置
        high_score_seatId = get_high_score_seatId(info['players_id'], info['high_score_player_id'])
        # 获得进行位置调整后的庄家位置
        dealer_flag = get_dealer(zhuang_id, high_score_seatId)
        battle_info = info['battle_info']
        # 遍历一个json文件中battle_info的所有动作
        for i in range(len(battle_info)):
            bat_info = battle_info[i]  # 当前动作信息
            # 当前动作是胡牌 且 当前玩家是高手玩家
            if bat_info['action_type'] == 'A' and bat_info['seat_id'] == high_score_seatId:
                # 获取当前动作的手牌信息
                handCards = changeSeat(bat_info['handcards'], high_score_seatId)  # 将高手玩家的牌调整到第一位
                # 获取高手玩家的手牌：
                handCards0 = handCards[0]
                #   将数据写入json文件
                fulu_ = changeSeat(bat_info['discards_op'], high_score_seatId)
                discards = bat_info['discards']
                discards_seq = changeSeat(bat_info['discards_real'], high_score_seatId)
                remain_card_num = get_remain_card(discards, fulu_, handCards)
                self_kingNum = self_king_num(king_card, handCards[0])  # 高手玩家的宝牌数
                fei_king_nums = get_feiKing_num(discards_seq, king_card)
                round_ = bat_info['round']
                storeFile = store_path + 'hu/' + f'hu{file_num_h}.json'
                dir_name = os.path.dirname(storeFile)
                # 如果文件夹不存在则创建文件夹
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                storeDict = {}
                storeDict['handCards0'] = handCards0
                storeDict['handCards'] = handCards
                storeDict['fulu_'] = fulu_
                storeDict['king_card'] = king_card
                storeDict['discards_seq'] = discards_seq
                storeDict['remain_card_num'] = remain_card_num
                storeDict['self_king_num'] = self_kingNum
                storeDict['fei_king_nums'] = fei_king_nums
                storeDict['discards'] = discards
                storeDict['round_'] = round_
                storeDict['dealer_flag'] = dealer_flag
                storeDict['isHu'] = 1  # 胡牌为1
                with open(storeFile, 'w', encoding="utf-8") as fp:
                    json.dump(storeDict, fp, indent=4)
                    file_num_h += 1
                print("ishu", file_num_h)
            # 当前动作为摸牌 且 摸牌后向听数为0 且 且下一个动作不是胡牌 且 当前玩家是高手玩家
            elif bat_info['action_type'] == 'G' and i < len(battle_info) - 1 and battle_info[i + 1][
                'action_type'] != 'A' and bat_info[
                'seat_id'] == high_score_seatId:
                # 将高手玩家的牌调整到第一位
                handCards = changeSeat(bat_info['handcards'], high_score_seatId)
                # 获取高手玩家的手牌：
                handCards0 = handCards[0]
                # 高手玩家的副露调整到第一位
                fulu_ = changeSeat(bat_info['discards_op'], high_score_seatId)
                catch_card = bat_info['operate_card']
                handCards0.append(catch_card)
                # 计算向听数是否为0
                xts1 = wait_types_comm_king(handCards0, fulu_[0], king_card)
                xts2 = wait_types_7(handCards0, fulu_[0], king_card)
                # xts3 = wait_types_haohua7(handCards0, fulu_[0], king_card)
                xts4 = wait_types_13(handCards0, fulu_[0], king_card)
                xts5 = wait_types_19(handCards0, fulu_[0], king_card)
                if min(xts1, xts2, xts4, xts5) == 0:
                    #   将数据写入json文件
                    discards = bat_info['discards']
                    discards_seq = changeSeat(bat_info['discards_real'], high_score_seatId)
                    remain_card_num = get_remain_card(discards, fulu_, handCards)
                    self_kingNum = self_king_num(king_card, handCards[0])  # 高手玩家的宝牌数
                    fei_king_nums = get_feiKing_num(discards_seq, king_card)
                    round_ = bat_info['round']
                    storeFile = store_path + 'nhu/' + f'nhu{file_num_nh}.json'
                    dir_name = os.path.dirname(storeFile)
                    # 如果文件夹不存在则创建文件夹
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    storeDict = {}
                    storeDict['handCards0'] = handCards0
                    storeDict['handCards'] = handCards
                    storeDict['fulu_'] = fulu_
                    storeDict['king_card'] = king_card
                    storeDict['discards_seq'] = discards_seq
                    storeDict['remain_card_num'] = remain_card_num
                    storeDict['self_king_num'] = self_kingNum
                    storeDict['fei_king_nums'] = fei_king_nums
                    storeDict['discards'] = discards
                    storeDict['round_'] = round_
                    storeDict['dealer_flag'] = dealer_flag
                    storeDict['isHu'] = 0  # 弃胡为0
                    with open(storeFile, 'w', encoding="utf-8") as fp:
                        json.dump(storeDict, fp, indent=4)
                        file_num_nh += 1
                    print("ishu", file_num_nh)

    print('文件数：不胡', file_num_nh, '->胡', file_num_h)


if __name__ == '__main__':
    file_num_nh = 0
    file_num_h = 0
    file_path = '/home/zonst/wjh/srmj/data/all/2/'
    store_path_hu = '/home/zonst/wjh/srmj/data/all/output/'
    for f in os.listdir(file_path):
        get_json_info_hu(file_path + f,store_path_hu)



