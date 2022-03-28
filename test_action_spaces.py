import action_space_builder

if __name__ == '__main__':
    builder = action_space_builder.CollectorActionSpaceBuilder()
    runs = 1000000
    found_action_spaces = set()
    for i in range(runs):
        if (i % 1000) == 0:
            print(i)
        plr_action_set = sorted(builder.build_player_action_space())
        npc_action_set = sorted(builder.build_npc_action_space())
        byte_repr = bytes(str(plr_action_set) + str(npc_action_set), 'utf-8')
        found_action_spaces.add(hash(byte_repr))
        
    print('Found {} unique action spaces out of {} runs'.format(len(found_action_spaces), runs))
