
export interface DataItem {
    content: string;
    createAt: number;
    extra: object; // 这里可以根据 extra 的具体结构定义一个更具体的类型
    id: string;
    meta: {
      avatar: string; // 这里假设 avatar 是一个 URL 字符串或一个表情符号
      title: string;
      backgroundColor?: string; // 可选属性，如果有的话
    };
    role: 'user' | 'assistant'; // 角色是 'user' 或 'assistant'
    updateAt: number;
  }