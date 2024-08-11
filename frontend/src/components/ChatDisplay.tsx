// components/ChatDisplay.tsx
import { ActionsBar, ChatList, ChatListProps, useControls, useCreateStore } from '@lobehub/ui';
import { data } from '../data';

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

  return (
    <div style={{ 
      position: 'absolute',
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      overflowY: 'auto', 
      padding: '10px', 
      
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

