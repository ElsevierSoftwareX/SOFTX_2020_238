#
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = 'Qi Chu <qi.chu@ligo.org>'
__date__ = '$Date$'
__cvs_tag__ = '$Name$'
try:
  __version__ = __cvs_tag__.split('-')[1] + '.' + __cvs_tag__.split('-')[2][0:-2]
except IndexError:
  __version__ = ''

