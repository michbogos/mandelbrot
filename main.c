#include <raylib.h>
#include <raymath.h>

int main(void)
{
    InitWindow(800, 450, "raylib [core] example - basic window");
    Shader FractalShader = LoadShader(0, "mandelbrot.fs");

    int iresLoc        = GetShaderLocation(FractalShader, "resolution");
    int topLeftloc     = GetShaderLocation(FractalShader, "topLeft");
    int bottomRightloc = GetShaderLocation(FractalShader, "bottomRight");

    

    float resolution[2] = {800, 450};

    float topLeft[2] = {0, 0};
    float bottomRight[2] = {1, 1};

    SetShaderValue(FractalShader, iresLoc, resolution, SHADER_UNIFORM_VEC2);

    while (!WindowShouldClose())
    {

        SetShaderValue(FractalShader, topLeftloc, topLeft, SHADER_ATTRIB_VEC2);
        SetShaderValue(FractalShader, bottomRightloc, bottomRight, SHADER_ATTRIB_VEC2);

        BeginDrawing();
            ClearBackground(BLACK);
            BeginShaderMode(FractalShader);
                    DrawRectangle(0, 0, 800, 450, BLACK);
            EndShaderMode();
            DrawFPS(10, 10);
        EndDrawing();

        if(IsMouseButtonDown(MOUSE_BUTTON_LEFT)){
            float mousex = ((float)GetMouseX()/800.0f);
            float mousey = ((float)GetMouseY()/450.0f);
            topLeft[0] += (bottomRight[0]-topLeft[0])*mousex*GetFrameTime();
            topLeft[1] += (bottomRight[1]-topLeft[1])*mousey*GetFrameTime();
            bottomRight[0] -= (bottomRight[0]-topLeft[0])*mousex*GetFrameTime();
            bottomRight[1] -= (bottomRight[1]-topLeft[1])*mousey*GetFrameTime();
        }
    }
    UnloadShader(FractalShader);
    CloseWindow();

    return 0;
}