// components/Header.tsx
import { ChatHeader } from '@lobehub/ui';
import { colors } from '@lobehub/ui';
import { ThemeSwitch } from '@lobehub/ui';
import { type ThemeMode } from 'antd-style';
import { useState } from 'react';


const Header = () => {
  const [themeMode, setThemeMode] = useState<ThemeMode>('auto');

  return (
    <header style={{ padding: '0px', backgroundColor: colors.geekblue.darkA[10], minHeight: '60px' }}>
      <ChatHeader
        showBackButton={false}
        height={"auto"}  
        color={colors.geekblue.dark[3]}
        right={
          <ThemeSwitch onThemeSwitch={setThemeMode} themeMode={themeMode} />
        }      
      />
      
    </header>
  );
};

export default Header;
