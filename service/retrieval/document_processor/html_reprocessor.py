from bs4 import BeautifulSoup
import os
from pathlib import Path

from bs4 import BeautifulSoup
import os
from pathlib import Path
import re

def clean_html_file(html_content, output_file_path=None, merge_columns=False):
    """
    HTML 파일을 읽어서 스타일을 제거하고 텍스트와 표만 남긴 후 저장
    merge_columns: True면 테이블 컬럼을 합쳐서 "컬럼명 값" 형태로 변환
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 테이블 컬럼 합치기 처리 (다른 정리 작업 전에 수행)
        if merge_columns:
            print("📊 테이블 컬럼 합치는 중...")
            merge_table_columns(soup)
        
        # 1. 불필요한 태그들 완전 제거
        print("🧹 불필요한 태그 제거 중...")
        unwanted_tags = ['script', 'style', 'link', 'meta', 'noscript', 'iframe', 'img']
        for tag_name in unwanted_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        
        # 2. 모든 태그의 스타일 관련 속성 제거
        print("✨ 스타일 속성 제거 중...")
        for tag in soup.find_all(True):
            attrs_to_remove = ['style', 'class', 'id', 'width', 'height', 
                             'bgcolor', 'color', 'font-family', 'font-size',
                             'margin', 'padding', 'border', 'background', 'face', 'size', 'align','lang']
            
            for attr in attrs_to_remove:
                if tag.has_attr(attr):
                    del tag[attr]
        
        # 3. 빈 태그들 제거
        print("🗑️  빈 태그 제거 중...")
        for tag in soup.find_all():
            if (not tag.get_text(strip=True) and 
                not tag.find_all() and 
                tag.name not in ['br', 'hr', 'img']):
                tag.decompose()
        
        # 4. 불필요한 서식 태그만 제거 (공백은 보존)
        for tag_name in ['font', 'u', 'b']:
            for tag in soup.find_all(tag_name):
                tag.unwrap()  # 태그는 제거하되 내용은 보존
        
        # 5. HTML을 문자열로 변환 (prettify 사용하지 않음)
        cleaned_html = str(soup)
        
        # 6. 연속된 공백만 정리 (단일 공백은 보존)
        cleaned_html = re.sub(r'\s+', ' ', cleaned_html)
        cleaned_html = re.sub(r'>\s+<', '><', cleaned_html)  # 태그 사이 공백만 제거
        cleaned_html = cleaned_html.replace('<p>', '').replace('</p>', '').replace('</span>', '').replace('<span>', '')    # 빈 <p> 태그 제거
        
        return cleaned_html
    
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        return None

def merge_table_columns(soup):
    """
    테이블의 각 행에서 컬럼명과 값을 합쳐서 "컬럼명 값" 형태로 변환
    """
    tables = soup.find_all('table')
    
    for table in tables:
        # 헤더 행 찾기 (첫 번째 tr 또는 thead 안의 tr)
        header_row = None
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
        else:
            # thead가 없으면 첫 번째 tr을 헤더로 간주
            first_tr = table.find('tr')
            if first_tr:
                header_row = first_tr
        
        if not header_row:
            continue
            
        # 헤더 텍스트 추출
        headers = []
        for cell in header_row.find_all(['th', 'td']):
            header_text = cell.get_text(strip=True)
            headers.append(header_text)
        
        if not headers:
            continue
        
        # 데이터 행들 처리
        tbody = table.find('tbody')
        if tbody:
            data_rows = tbody.find_all('tr')
        else:
            # tbody가 없으면 헤더 다음 행들을 데이터로 간주
            all_rows = table.find_all('tr')
            data_rows = all_rows[1:] if len(all_rows) > 1 else []
        
        for row in data_rows:
            cells = row.find_all(['td', 'th'])
            
            # 새로운 셀 내용 생성 (컬럼명 + 값)
            new_cells_content = []
            for i, cell in enumerate(cells):
                if i < len(headers):
                    header = headers[i]
                    value = cell.get_text(strip=True)
                    if value:  # 값이 있을 때만 합치기
                        merged_content = f"{header} {value}"
                    else:
                        merged_content = header
                    new_cells_content.append(merged_content)
            
            # 기존 셀들을 모두 제거하고 합친 내용으로 새 셀 하나 생성
            for cell in cells:
                cell.decompose()
            
            if new_cells_content:
                # 합친 내용을 하나의 셀로 만들기
                merged_text = " ".join(new_cells_content)
                new_cell = soup.new_tag('td')
                new_cell.string = merged_text
                row.append(new_cell)