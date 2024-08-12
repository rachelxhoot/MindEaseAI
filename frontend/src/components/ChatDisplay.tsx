import { ActionsBar, ChatList, ChatListProps, useControls, useCreateStore } from '@lobehub/ui';
import { data } from '../data';
import { GaussianBackground } from '@lobehub/ui';
import { useTheme } from 'antd-style';
import React from 'react';

const ChatDisplay = () => {
  const store = useCreateStore();
  const control: ChatListProps| any = useControls(
    {
      showTitle: false,
      type: {
        options: ['doc', 'chat'],
        value: 'doc',
      },
    },
    { store }
  );

  const theme = useTheme();

  const Options = useControls(
    'Options',
    {
      blurRadius: 16,
      fpsCap: 60,
      scale: 16,
    },
    { store },
  );
  const Layer1 = useControls(
    'Layer1',
    {
      color: theme.gold,
    },
    { store },
  );
  const Layer2 = useControls(
    'Layer2',
    {
      color: theme.cyan,
      maxVelocity: 0.2,
      orbs: 4,
      radius: 8,
    },
    { store },
  );
  const Layer3 = useControls(
    'Layer3',
    {
      color: theme.purple,
      maxVelocity: 0.2,
      orbs: 4,
      radius: 16,
    },
    { store },
  );

  return (
    <div style={{ 
      position: 'relative',
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      overflowY: 'auto', 
      padding: '10px',
      height: '100%',
      width: '100%',
    }}>
      <GaussianBackground
        layers={[Layer3, Layer2, Layer1]}
        options={Options}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: -1, // 确保背景层在所有其他元素之下
        }}
      />
      <ChatList
        data={data}
        renderActions={ActionsBar}
        renderMessages={{
          default: ({ id, editableContent }) => <div key={id}>{editableContent}</div>,
        }}
        style={{ width: '100%', height: '100%' }}
        {...control}
      />
    </div>
  );
};

export default ChatDisplay;