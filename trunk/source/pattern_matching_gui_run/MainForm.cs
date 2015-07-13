using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
//using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Threading;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace cudaForms
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
            Load += new EventHandler(Form1_Load);
            Activated += new EventHandler(Form1_Activated);
            
        }

        void Form1_Activated(object sender, EventArgs e)
        {

        }

        void Form1_Load(object sender, EventArgs e)
        {
            sbCC_ValueChanged(sender, e);
            sbFiltrationScale.Value = 25;
            loadMethods();
            jThumbnailView1.CanLoad = true;

        }

        private uint toLumen(Color _color)
        {
            if (_color.ToArgb() == -1)
                return 0;
            //return (uint)_color.ToArgb();
                /*
            return ((byte)(_color.R / 16) << 8) + ((byte)(_color.G / 16) << 4) + ((byte)_color.B / 16); */
            return (byte)((_color.R *_color.G*_color.B)/65025);
            //return (uint)((_color.R << 16) + (_color.B << 8) + _color.G);
            //return _color.T

        }

        public double CorrCoeff
        {
            get
            {
                return (double) (sbCC.Value)/100;
            }
            set { }
        }

        private void correlateTest()
        {
            clearRegions();
            double corrCoeff = CorrCoeff;
            Bitmap patternBitmap = new Bitmap(patternBox.Image);
            Bitmap featureBitmap = new Bitmap(originalImageBox.Image);

            int patternWidth = patternBitmap.Width;
            int patternHeight = patternBitmap.Height;

            uint[,] result = new uint[featureBitmap.Width, featureBitmap.Height];

            for(int y = 0; y < featureBitmap.Height; ++y)
            {
                for(int x = 0; x < featureBitmap.Width; ++x)
                {
                    result[x, y] = toLumen(featureBitmap.GetPixel(x, y));
                }
            }

            uint[,] patternData = new uint[patternWidth, patternHeight];

            for (int y = 0; y < patternBitmap.Height; ++y)
            {
                for (int x = 0; x < patternBitmap.Width; ++x)
                {
                    patternData[x, y] = toLumen(patternBitmap.GetPixel(x, y));
                }
            }

            uint patternEnergy = 0;
            for (int x = 0; x < patternWidth; ++x)
            {
                for (int y = 0; y < patternHeight; ++y)
                {
                    patternEnergy += (patternData[x, y])*(patternData[x, y]);
                }
            }

            patternEnergy = (uint)Math.Sqrt(patternEnergy);


            int[,] featureEnergy = new int[featureBitmap.Width, featureBitmap.Height];

            for (int x = 0; x < featureBitmap.Width - patternWidth; ++x)
                for (int y = 0; y < featureBitmap.Height - patternHeight; ++y)
                {
                    double sum = 0;
                    for (int px = 0; px < patternWidth; ++px)
                    {
                        for (int py = 0; py < patternHeight; ++py)
                        {
                            //sum += (result[x + px, y + py] - sumX[x, y]) * (result[x + px, y + py] - sumX[x, y]);
                            sum += result[x + px, y + py] * result[x + px, y + py];
                        }
                    }
                    featureEnergy[x, y] = (int) Math.Sqrt(sum);
                }

            double[,] corrRes = new double[featureBitmap.Width - patternWidth, featureBitmap.Height - patternHeight];


            for (int x = 0; x < featureBitmap.Width - patternWidth; ++x)
                for (int y = 0; y < featureBitmap.Height - patternHeight; ++y)
                {
                    double res = 0;
                    for(int px = 0; px < patternWidth; ++px)
                    {
                        for(int py = 0; py < patternHeight; ++py)
                        {
                            res += (patternData[px, py] * (result[x + px, y + py]));
                        }
                    }
                    res = res / (patternEnergy * featureEnergy[x, y]);
                    if (res > corrCoeff)
                    {
                        selectRegion(originalImageBox, new Rectangle(x, y, patternWidth, patternHeight));
                    }


                    originalImageBox.Refresh();
                    //featureBitmap.SetPixel(x, y, Color.FromArgb((int)color));
                }


/*            for (int x = 0; x < featureBitmap.Width - patternWidth; ++x)
                for (int y = 0; y < featureBitmap.Height - patternHeight; ++y)
                {
                    if (corrRes[x,y] > CorrCoeff)
                    {
                        selectRegion(pictureBox1, new Rectangle(x, y, patternWidth, patternHeight));
                        //uint color = 0;
                        //color = (uint)((double)((uint)Color.White.ToArgb()) * corrRes[x, y]);
                    }
                } */
            //pictureBox1.Image = featureBitmap;

        }

        private void button3_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.SelectedPath = (new DirectoryInfo(".")).FullName;
            if( folderBrowserDialog1.ShowDialog() == DialogResult.OK)
            {
                jThumbnailView1.FolderName = folderBrowserDialog1.SelectedPath;
                
            }
        }


        private void jThumbnailView1_DoubleClick(object sender, EventArgs e)
        {
            if(jThumbnailView1.SelectedItems.Count > 0)
            {
                string s = jThumbnailView1.SelectedItems[0].Tag.ToString();
                loadImage(originalImageBox, s);
                
            }
        }

        void loadImage(PictureBox _pictureBox, string _path)
        {
            _pictureBox.Image = Image.FromFile(_path);
            _pictureBox.Width = _pictureBox.Image.Width;
            _pictureBox.Height = _pictureBox.Image.Height;
            logItem("Загружено изображение: " + _path);
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            sbCC.Enabled = (cbMatch.SelectedItem as MethodDesc).CoeffEnabled;
            label2.Enabled = sbCC.Enabled;
        }

        private void clearRegions()
        {
            m_Results.Clear();
           
        }

        private void selectRegion(PictureBox _picture, Rectangle _coords)
        {
            m_Results.Add(_coords);
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            //selectRegion(pictureBox1, new Rectangle(30, 30, 100, 100));
            //selectRegion(pictureBox1, new Rectangle(80, 80, 100, 100));
            correlateTest();
        }

        private void sbCC_ValueChanged(object sender, EventArgs e)
        {
            label2.Text = CorrCoeff.ToString();
        }

        private void loadMethods()
        {
            cbMatch.Items.Add(new MethodDesc("walsh_hadamard_run.exe", "CPU/Walsh-Hadamard (яркость)", "{0} {1} {2} {3} lumen", false));
            cbMatch.Items.Add(new MethodDesc("walsh_hadamard_run.exe", "CPU/Walsh-Hadamard (цветность)", "{0} {1} {2} {3} rgb", false));
            cbMatch.Items.Add(new MethodDesc("correlation_cuda_run.exe", "CUDA/Fast correlation (яркость)", "{0} {1} {2} {3} cuda", true));
            cbMatch.Items.Add(new MethodDesc("correlation_cuda_run.exe", "CPU/Fast correlation (яркость)", "{0} {1} {2} {3} cpu", true));
            cbMatch.Items.Add(new MethodDesc("simple_matching_run.exe", "CUDA/Simple matching (яркость)", "{0} {1} {2} {3} cuda", true));
            cbMatch.Items.Add(new MethodDesc("simple_matching_run.exe", "CPU/Simple matching (яркость)", "{0} {1} {2} {3} cpu", true));

            cbFilter.Items.Add(new MethodDesc("filtration_cuda_run.exe", "Sobel/CUDA", "{0} {1} sobel {2}", false));
            cbFilter.Items.Add(new MethodDesc("filtration_cuda_run.exe", "Puritt/CUDA", "{0} {1} puritt {2}", false));
            cbFilter.Items.Add(new MethodDesc("filtration_cuda_run.exe", "Laplasian/CUDA", "{0} {1} laplas {2}", true));
            cbFilter.Items.Add(new MethodDesc("filtration_run.exe", "Sobel", "{0} {1} sobel {2}", false));
            cbFilter.Items.Add(new MethodDesc("filtration_run.exe", "Puritt", "{0} {1} puritt {2}", false));
            cbFilter.Items.Add(new MethodDesc("filtration_run.exe", "Laplasian", "{0} {1} laplas {2}", true));

            cbFilter.SelectedIndex = 0;
            cbMatch.SelectedIndex = 0;
        }

        private void button5_Click(object sender, EventArgs e)
        {
            clearRegions();

            if (cbFilter.SelectedIndex == -1) return;

            logItem("Фильтрация оригинала");
            filterPictureBox(originalImageBox);
            logItem("Фильтрация изображения");
            filterPictureBox(patternBox);
        }

        private void filterPictureBox(PictureBox _box)
        {
            if (_box.Image == null)
            {
                return;
            }
            logItem("Фильтрация");
            logItem("Создание временного файла");
            string fileNameIn = Guid.NewGuid().ToString() + ".bmp";
            string fileNameOut = Guid.NewGuid().ToString() + ".bmp";
            logItem(fileNameIn);
            logItem(fileNameOut);
            try
            {
                File.Delete(fileNameIn);
                File.Delete(fileNameOut);
            }
            catch (System.Exception _e)
            {
                
            }

            Bitmap bitmap = new Bitmap(_box.Image);
            bitmap.Save(fileNameIn, System.Drawing.Imaging.ImageFormat.Bmp);

            string path = (cbFilter.SelectedItem as MethodDesc).FilePath;
            string args = (cbFilter.SelectedItem as MethodDesc).Args;

            ProcessStartInfo info = new ProcessStartInfo(path);
            info.Arguments = String.Format(args, fileNameIn, fileNameOut, sbFiltrationScale.Value);
            logItem("Вызов модуля фильтрации: " + info.FileName);
            Process converter = Process.Start(info);
            converter.WaitForExit();
            _box.Load(fileNameOut);

            logItem("Удаление временных файлов");
            File.Delete(fileNameIn);
            File.Delete(fileNameOut);

        }

        public class MethodDesc
        {
            public string FilePath;

            public string Caption;

            public string Args;

            public bool CoeffEnabled;

            public MethodDesc(string _path, string _caption, string _args, bool _cEnabled)
            {
                FilePath = _path;
                Caption = _caption;
                Args = _args;
                CoeffEnabled = _cEnabled;
            }

            public override string ToString()
            {
                return Caption;    
            }
        }

        private void jThumbnailView1_Click(object sender, EventArgs e)
        {

            contextMenuStrip1.Show(Cursor.Position);
        }
        static bool once = true;
        private void jThumbnailView1_OnLoadComplete(object sender, EventArgs e)
        {
            if (!once) return;
            once = false;
            DirectoryInfo info = new DirectoryInfo(".");
            jThumbnailView1.FolderName = info.FullName;

        }

        private void оригиналToolStripMenuItem_Click(object sender, EventArgs e)
        {
            clearRegions();
            clearLog();
            logItem("Загрузка оригинала");
            if (jThumbnailView1.SelectedItems.Count > 0)
            {
                string s = jThumbnailView1.SelectedItems[0].Tag.ToString();
                loadImage(originalImageBox, s);
                m_OriginalFileName = s;
            }
        }

        private void паттернToolStripMenuItem_Click(object sender, EventArgs e)
        {
            clearRegions();
            logItem("Загрузка шаблона");
            if (jThumbnailView1.SelectedItems.Count > 0)
            {
                string s = jThumbnailView1.SelectedItems[0].Tag.ToString();
                loadImage(patternBox, s);
                m_PatternFileName = s;
            }
        }

        string m_PatternFileName = null;
        string m_OriginalFileName = null;

        private void button2_Click(object sender, EventArgs e)
        {
            clearRegions();
            if (cbMatch.SelectedIndex == -1) return;

            if ((originalImageBox.Image == null) || (patternBox.Image == null))
            {
                MessageBox.Show("Не загружены изображения");
                return;
            }

            string path = (cbMatch.SelectedItem as MethodDesc).FilePath;
            string args = (cbMatch.SelectedItem as MethodDesc).Args;

            ProcessStartInfo info = new ProcessStartInfo(path);
            info.Arguments = String.Format(args, "work\\input.bmp", "work\\pattern.bmp", "result.txt", 
                sbCC.Value.ToString());

            Bitmap originalBitmap = new Bitmap(originalImageBox.Image);
            originalBitmap.Save("work\\input.bmp", System.Drawing.Imaging.ImageFormat.Bmp);

            Bitmap patternBitmap = new Bitmap(patternBox.Image);
            patternBitmap.Save("work\\pattern.bmp", System.Drawing.Imaging.ImageFormat.Bmp);

            info.WorkingDirectory = Application.StartupPath;
            DateTime startTime = DateTime.Now;
            Process process = Process.Start(info);
            process.WaitForExit();
            m_ProcessTime.Text = (DateTime.Now - startTime).ToString();

            StreamReader file = File.OpenText("result.txt");
            while (!file.EndOfStream)
            {
                string res = file.ReadLine();
                Regex regex = new Regex("(\\d+)\\s+(\\d+)\\s+(\\d+)");
                MatchCollection result = regex.Matches(res);
                foreach (Match match in result)
                {
                    int x = int.Parse(match.Groups[1].ToString());
                    int y = int.Parse(match.Groups[2].ToString());
                    int distance = int.Parse(match.Groups[3].ToString());
                    selectRegion(originalImageBox, new Rectangle(x, y, patternBox.Image.Width, patternBox.Image.Height));
                }
            }

            originalImageBox.Refresh();
            file.Close();
        }

        private void pictureBox1_Paint(object sender, PaintEventArgs e)
        {
            foreach (Rectangle r in m_Results)
            {
                Graphics dc = e.Graphics;
                
                dc.FillRectangle(new SolidBrush(Color.FromArgb(130, Color.LightBlue)), r);
                dc.DrawRectangle(new Pen(Color.FromArgb(200, Color.Black)), r);
            }
        }

        private void clearLog()
        {
            listBox1.Items.Clear();
        }

        private void logItem(string _item)
        {
            listBox1.Items.Add(_item);
        }

        private List<Rectangle> m_Results = new List<Rectangle>();

        private void button1_Click(object sender, EventArgs e)
        {
            clearRegions();
            clearLog();
            if( m_OriginalFileName != null )
                loadImage(originalImageBox, m_OriginalFileName);

            if( m_PatternFileName != null )
                loadImage(patternBox, m_PatternFileName);
        }

        private void hScrollBar1_ValueChanged(object sender, EventArgs e)
        {
            label6.Text = sbFiltrationScale.Value.ToString();
        }

        
    }
}
