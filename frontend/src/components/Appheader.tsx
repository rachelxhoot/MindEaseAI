import { ChatHeader } from '@lobehub/ui';
import { colors } from '@lobehub/ui';
import { ThemeSwitch } from '@lobehub/ui';
import { type ThemeMode } from 'antd-style';
import { useState } from 'react';
import { Header } from '@lobehub/ui';


interface HeaderProps {
  themeMode: ThemeMode;
  onThemeSwitch: (mode: ThemeMode) => void;
}

const Appheader: React.FC<HeaderProps> = ({ themeMode, onThemeSwitch }) => {
  return (
    <Header actions={<ThemeSwitch
      themeMode={themeMode}
      onThemeSwitch={onThemeSwitch}
    />} logo={'AI'} nav={'MindEase'} />

  );
};

export default Appheader;