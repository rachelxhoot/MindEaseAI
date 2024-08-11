// components/Header.js
import { ChatHeader } from '@lobehub/ui';
import { colors } from '@lobehub/ui';

const Header = () => {
  return (
    <header style={{ padding: '0px', backgroundColor: colors.geekblue.dark[8], minHeight: '60px' }}>
      <ChatHeader
        showBackButton={false}
        left={<div>Left</div>}
        right={<div>Right</div>}
      />
    </header>
  );
};

export default Header;
