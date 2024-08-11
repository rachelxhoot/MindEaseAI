// components/ChatInput.js
import { ActionIcon, ChatInputActionBar, ChatInputArea, ChatSendButton, TokenTag } from '@lobehub/ui';
import { Eraser, Languages } from 'lucide-react';
import { Flexbox } from 'react-layout-kit';

const ChatInput = () => {
  return (
    <div style={{ 
      padding: '10px', 
      backgroundColor: '#f5f5f5', 
      borderTop: '1px solid #ddd',
      height: '100px', // 设置一个固定高度，根据需要调整
    }}>
      <Flexbox style={{ height: '100%', position: 'relative' }}>
        <div style={{ flex: 1 }}></div>
        <ChatInputArea
          bottomAddons={<ChatSendButton />}
          topAddons={
            <ChatInputActionBar
              leftAddons={
                <>
                  <ActionIcon icon={Languages} />
                  <ActionIcon icon={Eraser} />
                  <TokenTag maxValue={5000} value={1000} />
                </>
              }
            />
          }
        />
      </Flexbox>
    </div>
  );
};

export default ChatInput;