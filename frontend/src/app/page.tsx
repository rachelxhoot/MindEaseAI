"use client";

import React, { useState } from 'react';
import { ThemeProvider } from '@lobehub/ui';
import Appheader from '@/components/Appheader';
import ChatDisplay from '@/components/ChatDisplay';
import ChatInput from '@/components/ChatInput';

import { type ThemeMode } from 'antd-style';
const Home = () => {
  const [themeMode, setThemeMode] = useState<ThemeMode>('auto');

  const handleThemeModeChange = (mode: ThemeMode) => {
    setThemeMode(mode);
  };

  return (
    <ThemeProvider themeMode={themeMode} onThemeModeChange={handleThemeModeChange}>
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <Appheader themeMode={themeMode} onThemeSwitch={handleThemeModeChange} />
        
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          <ChatDisplay />
        </div>
        <ChatInput />
      </div>
    </ThemeProvider>
  );
};

export default Home;