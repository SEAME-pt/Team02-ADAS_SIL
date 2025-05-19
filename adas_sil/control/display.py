import sys
import os
import pygame


class Display:
    def __init__(self, width=1280, height=720):
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ADAS Display")
        


    def update_display(self, rgb_surface=None, lane_surface=None, seg_surface=None, bev_surface=None, polylines_surface=None):
             
        # Clear display
        self.display.fill((0, 0, 0))
        
        # Display RGB image on left side
        if rgb_surface is not None:
            self.display.blit(rgb_surface, (0, 0))
        
        # Display lane mask on right side
        if lane_surface is not None:
            self.display.blit(lane_surface, (390, 0))
        
        # Display seg mask below
        if seg_surface is not None:
            self.display.blit(seg_surface, (0, 195))

        # if hasattr(self.detector, 'lane_points_vis'):
        #     lane_points_vis_rgb = cv2.cvtColor(self.detector.lane_points_vis, cv2.COLOR_BGR2RGB)
        #     lane_points_surface = pygame.surfarray.make_surface(lane_points_vis_rgb.swapaxes(0, 1))
        #     self.display.blit(lane_points_surface, (384, 480))
        

            # Display bird's eye view on bottom-right
        if bev_surface is not None:
            self.display.blit(self.bev_surface, (390, 195))
            
            # Add label
            font = pygame.font.SysFont('Arial', 24)
            text = font.render('Bird\'s Eye View', True, (255, 255, 255))
            self.display.blit(text, (780, 490))
            # Update the display

        # Display polylines visualization in a new position (could also replace one of the above)
        if hasattr(self, 'polylines_surface') and self.polylines_surface is not None:
            self.display.blit(self.polylines_surface, (780, 0))  # Adjust position as needed
            # Add label
            font = pygame.font.SysFont('Arial', 18)
            text = font.render('Lane Polylines', True, (255, 255, 255))
            self.display.blit(text, (780, 180))

        # Display current speed
        # if hasattr(self, 'current_speed'):
        #     speed_text = font.render(f"Speed: {self.current_speed:.1f} km/h", True, (255, 255, 255))
        #     self.display.blit(speed_text, (20, 60))

        pygame.display.flip()


    def cleanup(self):
       
        # quit pygame properly
        print("Quitting pygame...")
        pygame.display.quit()
        pygame.quit()
        
        print("Cleanup complete")