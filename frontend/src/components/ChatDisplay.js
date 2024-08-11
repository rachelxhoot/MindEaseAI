// components/ChatDisplay.js
import { ActionsBar, ChatList, useControls, useCreateStore } from '@lobehub/ui';
import { data } from '../data'; // 确保 data 文件路径正确

const ChatDisplay = () => {
  const store = useCreateStore();
  const control = useControls(
    {
      showTitle: false,
      type: {
        options: ['doc', 'chat'],
        value: 'chat',
      },
    },
    { store }
  );

  return (
    <div style={{ 
      position: 'absolute',
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      overflowY: 'auto', 
      padding: '10px', 
      backgroundColor: '#fff' 
    }}>
      <ChatList
        data={data}
        renderActions={ActionsBar}
        renderMessages={{
          default: ({ id, editableContent }) => <div key={id}>{editableContent}</div>,
        }}
        style={{ width: '100%' }}
        {...control}
      />
    </div>
  );
};

export default ChatDisplay;