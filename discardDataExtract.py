import json
import os


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


# 获取弃胡模型所需要的数据
def get_json_info_hu(data_dir, store_path):
    # get json file_names
    all_json = os.listdir(data_dir)
    file_num_nh = 0
    file_num_h = 0

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
        # 获取不碰时信息的步骤：首先获取每一个丢牌动作信息中的手牌信息(即高手玩家的手牌信息 和 当前丢的牌)，对高手玩家是否可以碰进行判断
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
            # 当前动作为摸牌 且 摸牌后向听数为0 且 且下一个动作不是胡牌 且 当前玩家是高手玩家
            elif (bat_info['action_type'] == 'G' and i < len(battle_info) - 1 and battle_info[i + 1][
                'action_type'] != 'A'
                  and bat_info['seat_id'] == high_score_seatId):
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
                xts3 = wait_types_haohua7(handCards0, fulu_[0], king_card)
                xts4 = wait_types_13(handCards0, fulu_[0], king_card)
                xts5 = wait_types_19(handCards0, fulu_[0], king_card)
                if min(xts1, xts2, xts3, xts4, xts5) == 0:
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

    print('文件数：不胡', file_num_nh, '->胡', file_num_h)


if __name__ == '__main__':
    for i in range(21):
        file_path = '/home/tonnn/.nas/wjh/dataset/' + str(i) + '/'
        store_path_pong = '/home/tonnn/.nas/wjh/dataset_proccessed/'
        get_json_info_hu(file_path, store_path_pong)
