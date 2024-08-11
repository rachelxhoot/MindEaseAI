"use client";
import Header from '../components/Header';
import ChatDisplay from '../components/ChatDisplay';  // 修正这里
import ChatInput from '../components/ChatInput';  // 修正这里

const Home = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Header />
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <ChatDisplay />
      </div>
      <ChatInput />
    </div>
  );
};

export default Home;