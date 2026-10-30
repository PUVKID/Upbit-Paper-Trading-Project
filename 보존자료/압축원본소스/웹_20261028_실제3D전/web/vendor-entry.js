import * as THREE from 'three';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';
import learningIllustration from './assets/learning-glass.png';
import {createIcons, Activity, ArrowDown, ArrowUp, ArrowRight, Check, ChevronDown,
  ChevronRight, CirclePause, CirclePlay, Database, Home, Layers, LayoutDashboard,
  RefreshCw, ShieldCheck, TrendingUp, Wallet, ReceiptText, Cpu, BarChart3, Info,
  Pause, Play, Eye, EyeOff, Timer, Sun, Moon, BookOpen, Flag, RotateCw} from 'lucide';

// CSP를 유지하며 스타일시트 적용 전에 저장된 테마를 설정합니다.
(() => {
  let theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  try {
    const stored = localStorage.getItem('upbit-theme');
    if (stored === 'light' || stored === 'dark') theme = stored;
  } catch (error) {
    document.documentElement.dataset.themeStorage = 'unavailable';
  }
  document.documentElement.dataset.theme = theme;
})();

window.UpbitVendors = {THREE, RoomEnvironment, createIcons, learningIllustration,
  icons: {Activity, ArrowDown, ArrowUp, ArrowRight, Check, ChevronDown, ChevronRight,
    CirclePause, CirclePlay, Database, Home, Layers, LayoutDashboard, RefreshCw,
    ShieldCheck, TrendingUp, Wallet, ReceiptText, Cpu, BarChart3, Info, Pause, Play,
    Eye, EyeOff, Timer, Sun, Moon, BookOpen, Flag, RotateCw}};
