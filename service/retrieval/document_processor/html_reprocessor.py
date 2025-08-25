from bs4 import BeautifulSoup
import os
from pathlib import Path

def clean_html_file(html_content, output_file_path=None):
    """
    HTML 파일을 읽어서 스타일을 제거하고 텍스트와 표만 남긴 후 저장
    """
    try:
        # BeautifulSoup으로 파싱
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. 불필요한 태그들 완전 제거
        print("🧹 불필요한 태그 제거 중...")
        unwanted_tags = ['script', 'style', 'link', 'meta', 'noscript', 'iframe', 'img']
        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # 2. 모든 태그의 스타일 관련 속성 제거
        print("✨ 스타일 속성 제거 중...")
        for tag in soup.find_all(True):
            # 스타일 관련 속성들 제거
            attrs_to_remove = ['style', 'class', 'id', 'width', 'height', 
                             'bgcolor', 'color', 'font-family', 'font-size',
                             'margin', 'padding', 'border', 'background', 'face', 'size', 'align','lang']
            
            for attr in attrs_to_remove:
                if tag.has_attr(attr):
                    del tag[attr]
        
        # 3. 빈 태그들 제거 (br, hr 등은 제외)
        print("🗑️  빈 태그 제거 중...")
        for tag in soup.find_all():
            if (not tag.get_text(strip=True) and 
                not tag.find_all() and 
                tag.name not in ['br', 'hr', 'img']):
                tag.decompose()
        
        # 4. 정리된 HTML을 문자열로 변환
        cleaned_html = soup.prettify()
        
        cleaned_html = cleaned_html.replace('\n', '').replace('\t', '').replace('<font>', '').replace('</font>', '').replace('<u>', '').replace('</u>', '').replace('<b>', '').replace('</b>', '').replace('  ','') # 줄바꿈과 탭 제거
        
        return cleaned_html
    
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return None